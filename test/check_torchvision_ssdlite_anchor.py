import torch
import torchvision

# 加载模型（不加载预训练权重）
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large()
model.eval()

# 获取 anchor 生成器（包含每个特征层的先验框数量信息）
anchor_generator = getattr(model, "anchor_generator", None)
if anchor_generator is None:
    anchor_generator = getattr(model.head.classification_head, "anchor_generator", None)

if anchor_generator is None:
    raise RuntimeError("无法从模型中获取 anchor_generator")

# 支持属性或可调用方法两种情况
num_anchors_per_location = getattr(anchor_generator, "num_anchors_per_location", None)
if num_anchors_per_location is None:
    # 备用：从 aspect_ratios 推断每层每个位置的先验框数
    if hasattr(anchor_generator, "aspect_ratios"):
        num_anchors_per_location = [len(a) for a in anchor_generator.aspect_ratios]
    else:
        raise RuntimeError("无法获取 num_anchors_per_location 或 aspect_ratios")
elif callable(num_anchors_per_location):
    num_anchors_per_location = num_anchors_per_location()

print("每个特征层的每个位置先验框数量:", num_anchors_per_location)

# 获取分类头的 ModuleList（兼容不同实现）
classification_head = model.head.classification_head
if hasattr(classification_head, "module_list"):
    cls_heads = classification_head.module_list
elif hasattr(classification_head, "cls_logits"):
    cls_heads = classification_head.cls_logits
else:
    raise RuntimeError("找不到分类头的 module_list 或 cls_logits")

num_layers = len(cls_heads)
print("特征图层数:", num_layers)

# 兼容性：从模块中查找第一个 Conv2d 的 out_channels（支持 Sequential、Module、Conv2d 等）
def _get_out_channels(mod):
    if hasattr(mod, "out_channels"):
        return mod.out_channels
    for m in mod.modules():
        if isinstance(m, torch.nn.Conv2d) and hasattr(m, "out_channels"):
            return m.out_channels
    raise RuntimeError("无法从分类头模块中获取 out_channels")

out_channels = _get_out_channels(cls_heads[0])
num_classes = out_channels // num_anchors_per_location[0]
print("类别数（含背景）:", num_classes)

# 构造虚拟输入
dummy_input = torch.randn(1, 3, 320, 320)

# 提取所有预测特征图
with torch.no_grad():
    features = model.backbone(dummy_input)
    feature_tensors = list(features.values())  # 顺序应与 num_anchors_per_location 一致

    total_anchors = 0
    for i, (feat, anchors_per_cell) in enumerate(zip(feature_tensors, num_anchors_per_location)):
        _, _, H, W = feat.shape
        layer_total = H * W * anchors_per_cell
        print(f"特征层 {i}: 尺寸 = {H} x {W}, 每个位置先验框数 = {anchors_per_cell}, 该层总数 = {layer_total}")
        total_anchors += layer_total

    print(f"\n总的先验框数量: {total_anchors}")

# 检查推荐输入尺寸（若有 weights 元信息）
try:
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    print("推荐输入尺寸（weights.meta）:", weights.meta.get("size") or weights.meta.get("image_size"))
except Exception:
    print("未找到 weights 元信息，模型名:", model.__class__.__name__)

# 辅助：验证任意输入尺寸下 backbone 输出的特征图尺寸
def check_input_size(h, w):
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        feats = model.backbone(x)
    sizes = [tuple(t.shape[-2:]) for t in feats.values()]
    print(f"输入 {h}x{w} -> 特征图尺寸: {sizes}")

# 示例检查
check_input_size(320, 320)
