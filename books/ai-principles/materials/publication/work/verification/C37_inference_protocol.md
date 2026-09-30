# C37 已训文章模型从磁盘到本机服务：运行范围

本章承接现行 C19 从随机参数在 WikiText-2 raw-v1 的 600 篇训练文章上训练四轮的模型。原 `work/results/causal_lm_best.pt` SHA 为 `e6bb405f90d18d8e805d6af30a7d5d8eaf01ce1528f3e3f955821b776f5e339f`；C18 manifest/tokenizer SHA 及原数据身份见 `work/data/c37_serving/SOURCE.md`。原 C19 测试已在前章打开，C37没有新的语言能力盲测，没有更新任何参数。所引服务试验不是 GPU 通用吞吐榜，源代码统一在 `work/code/c37_*.py`。

## 导出与生成

`python work/code/c37_export_serving_weights.py` 检查原checkpoint、C18当前 manifest 与 tokenizer，只把51个模型状态张量、固定配置和来源写入 `work/data/c37_serving/c19_model_only.pt`；`export_receipt.json` 记录原59,231,395B、导出19,737,098B/SHA `d22a0799e70c070c29d292cbb527d748c5b28089c3ce88632fa990490c6cff7c`。逐张量与原件完全相同，真实训练块16位输出 logits 最大差0。少读的主要是 AdamW 两份历史数、RNG与恢复游标，不是量化或蒸馏。程序以 `torch.load(...,weights_only=True)` 装载固定权重，`eval()`和`inference_mode()`算推断，不执行反传/优化。

原 C19 `generate` 每步在当前词列最后最多128位上重新前向；禁止预测 PAD/BOS，`greedy` 取最大分数，另一选择取 top40 后按 `softmax(logits/0.8)` 抽样。输出 EOS 即结束，否则达到预定上限才停。C37缓存程序保留这两个选择和停止规则；随机抽样是否字节同还依赖随机状态/运算顺序，准确性对照主要核同一权重的 logits 和贪心 ID。

C19原生成代码在可用CUDA时套BF16自动混合精度，本章服务与两路缓存比较统一使用FP32；因此“同一权重、同一滑窗规则”的准确性核验是**本章FP32全前缀和FP32缓存之间**的核验，不保证与C19那次BF16采样记录每个编号逐位一致。

## 缓存正确性与窗口

`python work/code/c37_cache_verify.py --out work/results/c37_cache_exactness_first.json` 在RTX 5090 Laptop GPU、FP32推断分别用16/64/120/127/128位真token前缀，核逐层 K/V prefill 与原 C19 全前缀下一词分数；不越窗时继续单token解码，最大 logits 绝对差 `9.5367e-6`。短提示 `The city of` 的16个贪心ID两路相同，文字为反复的 “the first time”；120位起始再生16步时，缓存法重建7次，整列仍与原法逐ID相同。实验结果在上述JSON，当前实现 `work/code/c37_cached_decode.py`。

每层的 K/V 来自该层注意力前的活动，同一token下一层K/V在上一层已经读了左侧上下文后才形成。C19使用块内固定绝对正弦位置，原无缓存法滑窗后把保留的128个token从位置0重标。保留旧K/V的逐项值在这个定义下已经不再是新窗口的值；仅弹出最老一项不合法。作者故意做该错误操作，最大logit差`0.218415`。现行缓存法到128项时对最后128位重新prefill，故效率在长输出处会损失。四层×每层K/V两份×宽192×FP32四字节=每请求每token6144B，满128位786432B；只计张量载荷，不含PyTorch分配器/碎片/临时矩阵。`torch.cat`逐步追加，非分页缓存、非跨请求共享。

后来为读者练习增加可选温度后，现行缓存源码再以新路径 `work/results/c37_cache_exactness_after_temperature.json` 重跑同一正确性核验；最大差仍 `9.5367e-6`、错误弹旧K/V差 `0.218415`、短长贪心ID同且长路重建7次。该复跑不产生新语言测试，只确认温度接口改动未破坏缓存比较。

## 延迟、吞吐和调度的区别

`python work/code/c37_latency_bench.py --out work/results/c37_latency_with_components.json` 在同GPU、同固定模型FP32、`torch.inference_mode`下计时；每段GPU同步前后、3次预热后7次重复取中位，单段分解另做21重复。强制生成16步以便对照，不因EOS提前退出。16/64/120提示的全前缀16步中位约43.75/43.74/50.54ms，缓存约30.69/31.10/33.14ms；120项那支16步中有7次重新prefill。单独prefill的三提示缓存中位约1.85/1.96/1.98ms；已保存K/V后的下一项计算约1.73/1.30/1.70ms，对重算全前缀下一项约2.54/2.06/2.55ms。这些毫秒读数受Python调用、GPU钟频和其它活动影响，含小模型计算路径，不分离纯GPU核时；不能把其中一项差解释成一般大模型公式保证。

同一离线程序另使8条各16位提示**同时到达、长度相同**，每条生成16位，共128输出位。依次单条缓存中位256.998ms/总498输出token/s，一次显式批中位29.434ms/总4349输出token/s。批后的八条几乎同时结束；依次执行的最晚一条约257ms才结束。该微实验没有网络、真实到达分布、动态长度、EOS提前结束或自动合批调度，不能称连续批处理。其报告含每次样本波动，主文只保留理解吞吐与用户等待的必要数字。

本机 `work/code/c37_local_server.py` 只绑定 `127.0.0.1`，启动时只加载上述固定权重一次；`GET /health` 返回模型身份与上下文，`POST /generate` 提供贪心或固定种子的top40抽样，`POST /generate_batch` 只接收用户**显式提交**的1—8个等token长度、贪心请求。服务限制1—64个新token，超长输入取最后128位并返回删去多少位；无效长度/异长显式批返回HTTP400。handler可并发解析，但一个锁使无关推断串行，未实现从不同HTTP请求中自动凑批。

服务曾在本机端口53467实际启动，`work/code/c37_local_service_probe.py` 发请求记录 `work/results/c37_local_http_first.json` 和热身后 `c37_local_http_warm_recheck.json`。首次单请求约3178ms、随后显式预热请求约944ms，混有冷启动，不拿它们与热批作算法比较。热身后的单条12新token模型内部约43.99ms，八条显式批12步模型内部约37.76ms；HTTP客户端所见约55.52/63.86ms，二者口径不同。批的第一条与单条贪心ID相同，无效max和异长批确实返回400，长提示确实修剪115位。服务随后Ctrl-C停止，端口检查无监听。本机localhost不是公开部署，不宣称处理远程并发故障或工具授权。

作者冷读后将单条 `sample_top40` 的温度作可选请求字段，默认仍0.8，仅允许有限实数0.1—5.0；贪心显式给温度、超出范围的温度直接返回400。新码在本机端口53469重新启动，`work/results/c37_local_http_temperature_recheck.json` 确认同`The city of`/seed7/12步、温度0.8与1.0给不同编号/文字，温度0返回400，原单条/八条贪心第一项仍完全一致、其它无效输入仍按400处理。不同会话绝对耗时已变化，此次单条约24.35ms、八条约36.23ms，不把不同日期/不同请求顺序的毫秒差说成算法退步或进步。新服务再次Ctrl-C关闭，53469端口无监听。

## CPU动态量化只改变部分计算

`python work/code/c37_quantization_probe.py --out work/results/c37_quantization_first.json` 使用本机PyTorch2.11仍可调用但官方已建议新工作迁往torchao的 `torch.ao.quantization.quantize_dynamic`，把模型25个`nn.Linear`换成CPU动态INT8。Embedding、LayerNorm、softmax、KV活动等仍FP32；保存的可重新加载量化状态9,729,461B，比19,737,098B FP32服务包小，但元数据格式不完全相同。首64份现有C18验证块，共7944目标的平均NLL由FP32 `5.780105`到INT8 `5.780782`；三条固定提示的32步贪心序列有2条全同，`The city of`后来分叉。单进程CPU八线程、2次预热/5次重复，16新token全前缀生成的中位FP32 `14.07ms`，INT8 `31.77ms`，这台机器这条小模型路径反而更慢。不能把文件缩小推出端到端提速，也不能将此结果当SmoothQuant、GPU INT8或新量化栈成绩。

## 证据等级和剩余边界

导出张量相同、缓存窗口内近似数值相同/贪心同、错误弹出导致分数变化，是**特定实现的正确性核验**；耗时数字是**本机有限重复测量**，不是跨硬件服务研究；NLL来自先前已存在的验证块，不是新独立能力测试。C19弱自由生成在本章仍弱，HTTP接口不会教模型事实或工具行动。ORCA/PagedAttention/DistServe/FlashAttention等一手论文作为有条件的其它系统设计研究，均未在本机复刻。用户实际阅读、服务可用性验收与主模型能力提升未观察。
