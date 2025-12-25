# scrapephoto 包说明

该脚本包目的是构建复现CEO NARCISSISM MEASUREMENT AND IMPACT该论文的CEO人像部分提取、验证并赋分 CEO 照片的完整工具集。前期已完整抓取美国上市公司面板Def 14 HTML文件，总计5万＋。

## 核心脚本（最小必要集）
- `ceo_photo_pipeline_test.py`：主流水线。抓取 HTML、定位候选图片、OpenCV 粗检、DashScope 复核、写入结果 JSON。
- `dashscope_vision.py`：DashScope Vision API 轻量封装（同步/异步、速率控制、并行对比）。
- `progress_utils.py`：进度与时间预估小工具。
- `build_ceo_photo_check.py`：读取 CSV + 照片池，重新赋分并生成 Excel 报告（可选嵌入缩略图）。
- `check_dashscope_health.py`：DashScope 可用性探针。
- 常用辅助：
`dashscope_api_key.txt`（本地 API key 占位）、`execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv`（主数据源）。

## 目录分组
- 主流水线：`pipeline/ceo_photo_pipeline.py`、`pipeline/dashscope_vision.py`、`utils/progress_utils.py`
- 报告/数据：`reporting/build_ceo_photo_check.py`、`reporting/export_rows_with_photos.py`、`reporting/dedupe_photos.py`、`reporting/remove_excel_images.py`
- 反馈/学习：`learning/label_ceo_photos.py`、`learning/parse_train_data.py`、`learning/ceo_photo_pattern_learner.py`、`learning/apply_learned_patterns.py`、`learning/learned_optimization.py`、`learning/feedback_learning.py`、`learning/train_and_test.py`
- 诊断/健康：`diagnostics/check_dashscope_health.py`、`diagnostics/test_compare_faces.py`
- 参考文档：`COMPLETE_SYSTEM_GUIDE.txt`、`FEEDBACK_WORKFLOW.txt`、`MEMORY_SYSTEM_GUIDE.txt`、`README_PATTERN_LEARNING.txt`、`QUICK_START*.txt`、`GROUP_PHOTO_HANDLING.txt` 等
- 数据与密钥：`execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv`、`dashscope_api_key.txt`、`google_api_key.txt`、`google_cse_id.txt`
- 结果与日志（示例）：`ceo_photo_results.json`、`ceo_photo_pipeline.log`、`ceo_photo_check.json`


## 安装
```bash
pip install -e .
```
依赖：`pandas`、`requests`、`beautifulsoup4`、`pillow`、`openpyxl`、`opencv-python`。DashScope 不依赖官方 SDK，直接 HTTP 调用；只需在 `scrapephoto/dashscope_api_key.txt` 写入密钥。

## 快速使用
1) 准备数据与密钥  
   - 将 SEC 报表 HTML 解压到本地（如 `D:/sec_filings`）。  
   - `scrapephoto/dashscope_api_key.txt` 写入有效 key。  
   - 照片池目录默认 `D:/ceo_photo_pool`，可通过参数覆盖。

2) 运行主流水线（示例）  
```bash
python -m scrapephoto.pipeline.ceo_photo_pipeline \
  --csv execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv \
  --sec-root D:/sec_filings \
  --shared-output D:/ceo_photo_pool \
  --results-output scrapephoto/ceo_photo_results.json \
  --log-file scrapephoto/ceo_photo_pipeline.log \
  --dashscope-concise
```
常用调参：`--limit/--offset` 控制处理范围，`--strategy-limit` 限制策略数量，`--brute-force-strategy` 启用暴力兜底，`--dashscope-async` 切换异步批量校验。

3) 生成核对报告  
```bash
python scrapephoto/reporting/build_ceo_photo_check.py \
  --csv execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv \
  --photo-pool D:/ceo_photo_pool \
  --output scrapephoto/ceo_photo_check.xlsx \
  --insert-photos
```

4) 辅助维护
- API 健康检查：`python scrapephoto/diagnostics/check_dashscope_health.py --expect-person`

## 反馈/学习组件（机器学习）
- `label_ceo_photos.py` / `parse_train_data.py`：标注样本并转成训练格式。
- `ceo_photo_pattern_learner.py`：从标注样本学习常见位置/阶段模式。
- `apply_learned_patterns.py`：将学习到的模式用于候选重排或过滤。
- `learned_optimization.py`：汇总提取结果，输出可调参数摘要。
- `feedback_learning.py`：记录人工反馈，计算准确率、支撑持续改进。
- `train_and_test.py`：简单训练/评估骨架，可插入自定义特征与策略。

## 结果
- 统一照片输出目录：`--shared-output`（默认 `D:/ceo_photo_pool`）  
- 结果 JSON：`--results-output`（默认 `scrapephoto/ceo_photo_results.json`）  
- 日志：`--log-file`（默认 `scrapephoto/ceo_photo_pipeline.log`）  
- Excel 报告：`scrapephoto/ceo_photo_check.xlsx`（1和0，可嵌入缩略图）
