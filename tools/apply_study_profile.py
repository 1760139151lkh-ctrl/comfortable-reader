"""Merge a book-owned, inert presentation profile without touching its resources."""
from pathlib import Path
import argparse,json,hashlib

ACTIVITY_FIELDS={'presentation','result_suffix','reading_guide','practice','description','key_symbol','reference_asset','parameters','prediction','training_description'}
def apply_profile(base:Path,profile:Path,out:Path):
 if out.exists():raise ValueError('输出已存在，请保留旧版本并使用新路径')
 if base.stat().st_size>16*1024*1024 or profile.stat().st_size>2*1024*1024:raise ValueError('学习描述超过读取预算')
 pack=json.loads(base.read_text(encoding='utf-8'));spec=json.loads(profile.read_text(encoding='utf-8'))
 if spec.get('format')!='comfortable-study-presentation@1' or pack['book_uuid']!=spec['book_uuid'] or pack['book_revision_sha256']!=spec['epub_sha256']:raise ValueError('呈现说明与书籍身份或 EPUB 修订不一致')
 activities={a['id']:a for a in pack.get('activities',[])};assets={a['id']:a for a in pack['assets']}
 for key,fields in spec.get('activities',{}).items():
  if key not in activities or set(fields)-ACTIVITY_FIELDS:raise ValueError('活动或呈现字段没有登记：'+key)
  if fields.get('reference_asset') and fields['reference_asset'] not in assets:raise ValueError('参考记录没有登记：'+key)
  activities[key].update(fields)
 for key,fields in spec.get('assets',{}).items():
  if key not in assets or set(fields)-{'comparison_reference','media_identity','frame_descriptions'}:raise ValueError('资源或呈现字段没有登记：'+key)
  if fields.get('comparison_reference'):
   if fields['comparison_reference'] not in assets:raise ValueError('比较原件没有登记')
   assets[key]['comparison_reference']=fields['comparison_reference']
  if 'media_identity' in fields:assets[key].setdefault('media',{})['identity']=fields['media_identity']
  descriptions=fields.get('frame_descriptions',{})
  for frame in assets[key].get('media',{}).get('decoded_frames',[]):
   if frame['asset_id'] in descriptions:frame['description']=descriptions[frame['asset_id']]
 pack['presentation_version']=1
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(pack,ensure_ascii=False,indent=2),encoding='utf-8')
 return {'status':'presentation_merged','book_uuid':pack['book_uuid'],'epub_sha256':pack['book_revision_sha256'],'assets':len(assets),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'execution_authorized':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('--profile',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();print(json.dumps(apply_profile(a.base,a.profile,a.out),ensure_ascii=False))
