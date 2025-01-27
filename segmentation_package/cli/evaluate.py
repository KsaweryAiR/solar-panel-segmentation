import torch
import torch.onnx
import matplotlib.pyplot as plt
from segmentation_package.data_module.datamodulepanels import PanelsDataModule
from segmentation_package.models.segmentation import LitModel

if __name__ == "__main__":
    data_module = PanelsDataModule()
    data_module.setup()

    # Wybierz model
    checkpoint_path = "lightning_logs/version_19/checkpoints/epoch=170-val_loss=0.15-val_BinaryF1Score=0.85.ckpt"
    model = LitModel.load_from_checkpoint(checkpoint_path).eval().cpu()
    x = torch.rand(1, 3, 256, 256)
    _ = model(x)
    model_name = "model_ver8.onnx"
    torch.onnx.export(model,
                      x,
                      model_name,
                      export_params=True,
                      opset_version=15,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sample_idx = 28
    image, mask = data_module.test_dataset[sample_idx]

    image = image.to(device)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        output = torch.sigmoid(output)
        output = output.squeeze().cpu()

    mask = mask.squeeze().cpu()
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask.numpy(), cmap='gray')
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(output.numpy(), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
