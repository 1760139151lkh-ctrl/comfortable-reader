# C47开放大学学习记录的身份与用途

2026-09-25从[UCI第349号、原作者署名的OULAD镜像页](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset)取得原zip URL https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip，46,748,244字节/SHA-256 f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4；ZIP全成员CRC通过，成员名单白名单验证。只解 courses.csv、studentInfo.csv、studentRegistration.csv、studentVle.csv、OULAD.names，逐件字节/SHA在manifest.json。未解出studentAssessment等评分资料，也不把它们作特征。源自Open University 2013/14的七门匿名课程22次开课，UCI页和[原团队资料说明](https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/)都标CC BY4，引用Kuzilek、Hlosta、Zdrahal 2017 Scientific Data论文；本书仅本机教学分析，保留原归属。[原论文方法/资料说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC5704676/)可核32,593条学生-课程开课记录与每日VLE点击汇总10,655,280行，不是学生逐秒注意力或教师干预随机分派。

本章选择同一模块BBB的四次旧开课（2013B、2013J、2014B、2014J），人数不读结果标签先查分别1,767/2,237/1,613/2,292。前两期合做拟合、2014B做验证、2014J在方法固定后作后时评价；同一匿名id若重复出现在更早开课，后一行在分组时排除，避免一人训练/测试穿越。原团队警告B/J课程结构可能不同，所以跨开课差异是实际任务条件，不可把这项测试自动外推到别的学校或今天。最终标签只把原final_result的Pass/Distinction归1、Fail/Withdrawn归0；它是**该课程档案的最终类别**，不是真正“有无能力”或该不该被老师放弃的价值判断。

预测时点分别为开课后第7/21/42天，只允许在该时点已注册且没有**已知在当日及此前注销**的行进入。studentRegistration的缺值在原CSV写作问号，由to_numeric(errors=coerce)处理；若注册日未知，排除出本章观察人群，未知注销日视为没有记载注销。对于仍活跃的人，仅累计日期0至该时点前一日的每日sum_click，每7天合成一格，不读未来点击、未来最终结果或以后评卷分数作特征；原始匿名ID只供连接/去重，模型不使用ID或性别、年龄、地区、残障、既往学位等人口字段。点击数有平台记录选择、资源安排、学习者时间/设备与已获帮助的混合影响；它不等于理解程度。有人最初几周零点击，不等于没有在离线学习；同一天大量点击不等于认真学会。

这份发布物公开可取也不能证明任意现实学生干预已获授权。书中只报聚合计数/误差，不展示单个学生的原始行、不导出个人名单、不联系任何学生。按既定历史期训练出的模型不应用于今天任何人的录取、退出或教学资源决定。预先切分、评价与“预测不等于推荐有益”的任务边界在work/verification/C47_oulad_pretest_decision.md及后续C47协议。
