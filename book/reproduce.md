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

在远程服务器构建后，可用 `scp` 下载 `book/_build/html/`，或通过 SSH 端口转发
访问预览服务器：`ssh -L 8000:localhost:8000 USER@HOST`。

## 更新图片

如果重新生成了实验报告，在项目根目录运行：

```bash
cp outputs/runs/gpt12_babylm_clone/training_report.png book/assets/
cp outputs/runs/gpt12_babylm_clone/perplexity_report.png book/assets/
cp outputs/runs/gpt12_babylm_clone/blimp_sva_logit/blimp_report.png \
  book/assets/sva_logit_report.png
cp outputs/runs/gpt12_babylm_clone/blimp_conditional/blimp_report.png \
  book/assets/sva_conditional_report.png
```

数值表格是本次下载结果的快照。如果训练结果改变，也需要同步更新各 Markdown 页面的
汇总数值。
