# demark

去水印 CLI：图片走 OpenCV inpaint，视频走 ffmpeg `delogo`，覆盖四类日常场景——

| 场景 | 素材 | 水印形态 | 处理方式 |
|---|---|---|---|
| 视频会议截图 | 图片 | 角部名字条 / "录制中"角标 | TELEA/NS inpaint |
| 网络课程 | 视频 | 固定位置平台 logo | ffmpeg delogo |
| 网络视频 | 视频 | 固定/半透明角标（可多个） | ffmpeg delogo 多区域串联 |
| AI 生成图片 | 图片 | 角部平台徽标/文字 | inpaint + 角部自动检测 |

## 安装

```bash
pip install demark
```

系统依赖（必须单独安装）：ffmpeg/ffprobe。

```bash
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
```

从源码安装时锁定版本，**不要**引滚动的 `main` 分支/tarball（`pip install` 会隐性升级到未审阅代码）：

```bash
pip install https://github.com/helloworldtang/demark/archive/refs/tags/v0.1.0.tar.gz
```

## 快速上手

```bash
# 图片：自动检测角部水印，输出前后对比图目检
demark img meeting.png --auto --compare cmp.png

# 图片：手动指定区域（X,Y,W,H 像素坐标），或交互拖框
demark img shot.png -r 540,430,120,40
demark img shot.png --select

# 视频：自动检测静态叠层（抽 30 帧做时域分析）
demark video course.mp4 --auto

# 视频：先出首帧对比预览，目检无误再全量跑
demark video clip.mp4 -r 412,429,217,55 --preview-frame preview.png

# 干跑：只检测并打印区域 + 标注图，不处理
demark detect clip.mp4
```

输出默认写到 `<原文件名>.demark.<扩展名>`；`-o` 指定路径，图片支持 `--in-place`。

## 使用建议

- **先 `detect` 后处理**：自动检测是启发式的（宁漏勿滥），`demark detect` 会打印可直接复制使用的 `-r` 参数和标注预览图。
- **自动 + 手动可叠加**：`--auto` 与 `-r` 同时给时取并集。
- **视频质量**：默认 libx264 crf 18（近视觉无损），`--crf`/`--preset` 可调；音频轨直拷不动。
- **复杂背景上的文字水印**：图片路径的 `--method ns` 有时比 `telea` 重建得更自然。

## 工作机制

- **图片**：形态学梯度找文字/徽标状 blob，按面积和位置（四角色带 + 底部条带）过滤；修复用 OpenCV inpaint（TELEA/NS），mask 外扩 3px 吃掉抗锯齿边缘。
- **视频**：均匀抽 30 帧算逐像素时间标准差——水印永远不动，背景总会动；"时间稳定 × 边缘密集"的连通域即候选，再按面积/填充率过滤。去除用 ffmpeg `delogo` 滤镜（边缘插值），区域自动收缩到离帧边 ≥1px。

## 局限

- 只处理**位置固定**的水印；移动水印、全屏平铺水印不适用。
- delogo 是边缘插值，纯色/低频背景效果好，高频纹理背景会偏糊；图片路径的 inpaint 强得多。
- 自动检测对"整幅画面静止"的视频（如 PPT 静止页录屏）无能为力——此时用水印位置常识手动 `-r`。

## 开发

```bash
git clone https://github.com/helloworldtang/demark.git
cd demark
uv venv && uv pip install -e ".[dev]"   # 或 pip install -e ".[dev]"
pytest                                    # 测试（含 ffmpeg 合成素材 e2e）
ruff check src tests && mypy src          # 质量门
```

## License

MIT
