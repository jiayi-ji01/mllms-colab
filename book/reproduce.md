# 构建与更新

## 本地实时预览

在项目根目录执行：

```bash
python -m pip install -e '.[book]'
cd book
jupyter book start
```

命令运行后，在浏览器打开终端显示的地址，通常是 `http://localhost:3000`。

## 构建静态 HTML

```bash
cd book
jupyter book build --html --strict
python -m http.server 8000 --directory _build/html
```

浏览器打开 `http://localhost:8000`。静态网站位于 `book/_build/html/`，该生成目录
不会提交到 Git。

## 在 Colab 中构建

```python
%cd /content/mllms-colab
!pip install -e '.[book]'
%cd /content/mllms-colab/book
!jupyter book build --html --strict
```

Colab 不适合直接运行本地网站服务器。需要下载静态网站时，可以压缩：

```python
import shutil

shutil.make_archive(
    "/content/babylm_results_book",
    "zip",
    root_dir="/content/mllms-colab/book/_build/html",
)
```

## 更新图片

如果重新生成了实验报告，在项目根目录运行：

```bash
cp output/gpt12_babylm_clone_colab/training_report.png book/assets/
cp output/gpt12_babylm_clone_colab/perplexity_report.png book/assets/
cp output/gpt12_babylm_clone_colab/blimp_sva_logit/blimp_report.png \
  book/assets/sva_logit_report.png
cp output/gpt12_babylm_clone_colab/blimp_conditional/blimp_report.png \
  book/assets/sva_conditional_report.png
```

数值表格是本次下载结果的快照。如果训练结果改变，也需要同步更新各 Markdown 页面的
汇总数值。
