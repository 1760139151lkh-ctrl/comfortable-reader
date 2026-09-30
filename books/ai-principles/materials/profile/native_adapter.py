"""Versioned runtime adapter supplied with the AI-principles book.
Executed only after separate host review; not a reading-side script.
"""
import sys,json,runpy,traceback,platform,hashlib,os,importlib.util,ast,math

def run(request,root,entry):
    def load_module(name):
        spec=importlib.util.spec_from_file_location(name,entry)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    
    try:
        if request['activity_id']=='cached-request':
            import torch
            module=load_module('study_cached_request')
            model,tokenizer,state=module.load_serving(torch.device('cpu'))
            prompt=request['params'].get('prompt','Artificial intelligence')
            ids=[module.BOS]+tokenizer.encode(prompt).ids
            outputs={}
            for engine in ('full','cached'):
                result=module.generate(model,ids,12,engine)
                result['generated_text']=tokenizer.decode(result['new_token_ids'])
                outputs[engine]=result
            report={'scope':'one actual C19-model request; no training or network service','prompt':prompt,'outputs':outputs,'same_token_ids':outputs['full']['new_token_ids']==outputs['cached']['new_token_ids'],'model_sha256':hashlib.sha256(module.SERVING.read_bytes()).hexdigest()}
            (root/'work/results/cached_request.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False))
        elif request['activity_id']=='tool-request':
            import torch
            module=load_module('study_tool_request')
            data=root/'work/data/c38_tool_trajectories_v3'
            model,step,model_sha=module.load_model('trained',root/'work/runs/c38_tool_aligned_copy_2400/best.pt',torch.device('cpu'),data)
            tokenizer=module.Tokenizer.from_file(str(module.TOKENIZER))
            rows=[json.loads(line) for line in (data/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
            kind=request['params'].get('kind','multiply')
            row=next(row for row in rows if row['kind']==kind)
            catalog=json.loads((data/'title_catalog.json').read_text(encoding='utf-8'))
            report=module.evaluate_one(row,model,tokenizer,torch.device('cpu'),catalog,3)
            report.update(scope='one actual validation request from the published v3 action model; no copy gate; no host correction or training',model_sha256=model_sha,selected_step=step)
            (root/'work/results/tool_request.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False))
        elif request['activity_id']=='lm-train':
            import torch
            module=load_module('study_causal_train')
            if request['params']['device']=='cuda' and not torch.cuda.is_available():
                raise RuntimeError('所选 GPU 当前不可用；没有静默改用 CPU 训练。')
            module.select_output_dir(str(root/'work/results'))
            original_checkpoint=module.atomic_checkpoint
            def checkpoint(path,state):
                original_checkpoint(path,state)
                record={key:state.get(key) for key in ('epoch','next_batch','global_step','best_epoch','best_validation_nll','config','c18_manifest_sha256')}
                # Before the first validation the algorithm correctly uses +inf.
                # The UI receipt must still be strict JSON and available for resume.
                record={key:(None if isinstance(value,float) and not math.isfinite(value) else value) for key,value in record.items()}
                record.update(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity='actual algorithm checkpoint, not suspended Python stack')
                destination=root/'work/results/checkpoint-state.json'
                pending=destination.with_suffix('.writing')
                pending.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
                os.replace(pending,destination)
            module.atomic_checkpoint=checkpoint
            module.train(module.TrainConfig(epochs=4),bool(request['params'].get('resume_from')),None)
        elif request['activity_id']=='lm-probe':
            import importlib.util
            module=load_module('study_causal_probe')
            module.select_output_dir(str(root/'work/results'))
            # Retain the original probe and expose its actual batch separately.
            original_batch=module.batch
            def visible_batch(part,indices,device):
                values=original_batch(part,indices,device)
                ids,labels,real=values
                record={'input_ids':ids.detach().cpu().tolist(),'labels':labels.detach().cpu().tolist(),'attention_mask':real.detach().cpu().tolist(),'visibility':'position t can read positions 0..t; label t is the next token','indices':indices.tolist()}
                (root/'work/results/visible_batch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                return values
            module.batch=visible_batch
            module.probe()
        else:
            # A capped line trace records the actual execution's changing state.
            # It is a replay record, not an independent JavaScript implementation.
            trace=[]; before_state=None
            lines=entry.read_text(encoding='utf-8').splitlines()
            def function_signature(source,name):
                node=next((n for n in ast.parse(source,filename=str(entry)).body if isinstance(n,ast.FunctionDef) and n.name==name),None)
                return ast.dump(node,include_attributes=False) if node else None
            linear_contract=False
            if request['activity_id']=='perceptron':
                reference=(root/'reference-source.py').read_text(encoding='utf-8')
                current=entry.read_text(encoding='utf-8')
                linear_contract=all(function_signature(reference,n)==function_signature(current,n) for n in ('score','predict'))
                (root/'work/results/visualization-contract.json').write_text(json.dumps({'linear_score_matches_reference':linear_contract,'scope':'AST identity of score and predict; not a proof of arbitrary edited program semantics'}),encoding='utf-8')
            def observe(frame,event,arg):
                nonlocal before_state
                if request['activity_id']=='perceptron' and event=='line' and frame.f_code.co_filename==str(entry) and len(trace)<1600:
                    line=lines[frame.f_lineno-1].strip()
                    phase='before_rule' if line.startswith('if label * old_score') else 'after_update' if line=='updates += 1' else None
                    if phase:
                        v=frame.f_globals
                        try:
                            state={'phase':phase,'line':frame.f_lineno,'epoch':v['epoch'],'features':list(v['features']),'label':v['label'],'weights':list(v['weights']),'bias':v['bias'],'score_before':v['old_score']}
                            if phase=='before_rule':before_state=state.copy()
                            elif before_state:
                                state.update(old_weights=before_state['weights'],old_bias=before_state['bias'],weight_delta=[a-b for a,b in zip(state['weights'],before_state['weights'])],bias_delta=state['bias']-before_state['bias'])
                            # Unsupported custom objects must not alter or abort the program.
                            json.dumps(state);trace.append(state)
                        except (KeyError,TypeError,ValueError):pass
                return observe
            if request['activity_id']=='perceptron':sys.settrace(observe)
            try:runpy.run_path(str(entry),run_name='__main__')
            finally:
                sys.settrace(None)
                if trace:(root/'work/results/execution_trace.json').write_text(json.dumps(trace,ensure_ascii=False),encoding='utf-8')
    except BaseException:
        traceback.print_exc()
        raise
