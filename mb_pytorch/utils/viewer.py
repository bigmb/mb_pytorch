import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.functional as TF
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
import io
import PIL

__all__ = ['show_images', 'show_segmentation_masks', 'show_bounding_boxes', 'show_label_on_img','model_viewer','new_show_cam_on_image','gradcam_viewer','plot_classes_pred']

def show_images(imgs, figsize=(12.0, 12.0)):
    """Displays a single image or list of images. 
    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        figsize (Tuple[float, float]): size of figure to display.
    Returns:
        None
    """

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, img in enumerate(imgs):
        #img = img.detach()
        #img = TF.to_pil_image(img)
        img = np.asarray(img)
        if img.shape[0]==3:
            img = np.transpose(img, (1, 2, 0))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None
    
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    Source: (https://www.tensorflow.org/tensorboard/image_summaries)."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)  
    img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0      
    #img_ar = np.array(image)
    #image = torch.Tensor(img_ar)
    return img_tensor

def create_img_grid(data,labels, t_writer, n_images=9, g_i=3, g_j=3,global_step=0,gray_image=False):
    """
    Creates a grid of images and sends it to TensorBoard.
    Input:
        data: A list of images
        labels: A list of labels
        t_writer: TensorBoard writer
        n_images: Number of images to display
        g_i: Number of rows in the grid
        g_j: Number of columns in the grid
    """
    
    data_read = data[:n_images]
    labels_read = labels[:n_images]
    
    ### Visualize a 3x3 grid
    fig, axes = plt.subplots(g_i, g_j, figsize=(15,15),
            subplot_kw = {'xticks':[], 'yticks':[]},
            gridspec_kw = dict(hspace=0.3, wspace=0.01))
    
    if gray_image:
        for i, ax in enumerate(axes.flat):
            img_cur = np.array(data_read[i,0,:,:])
            ax.title.set_text(int(labels_read[i]))
            ax.imshow(img_cur, cmap='gray')
    else:
        for i, ax in enumerate(axes.flat):
            img_cur = np.array(data_read[i])
            img_cur = np.transpose(img_cur, (1, 2, 0))
            ax.title.set_text(int(labels_read[i]))
            ax.imshow(img_cur)
    
    ### Send the figure over to TensorBoard
    t_writer.add_image('grid', plot_to_image(fig), global_step=global_step)
    plt.close(fig)

    

def show_segmentation_masks(imgs, masks, figsize=(12.0, 12.0)):
    """Displays a single image or list of images with segmentation masks.
    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        masks (Union[List[torch.Tensor], torch.Tensor]): A list of masks
            of shape (1, H, W) or a single mask of shape (1, H, W).
        figsize (Tuple[float, float]): size of figure to display.
    Returns:
        None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(masks, list):
        masks = [masks]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        # img = img.detach()
        # img = TF.to_pil_image(img)
        # mask = mask.detach()
        # mask = TF.to_pil_image(mask)
        img = np.asarray(img)
        if img.shape[0]==3:
            img = np.transpose(img, (1, 2, 0))
        mask = np.asarray(mask)
        if mask.shape[0]==3:
            mask = np.transpose(mask, (1, 2, 0))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].imshow(np.asarray(mask), alpha=0.5)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None

def show_bounding_boxes(imgs, boxes, figsize=(12.0, 12.0)):
    """Displays a single image or list of images with bounding boxes.
    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        boxes (Union[List[torch.Tensor], torch.Tensor]): A list of boxes
            of shape (N, 4) or a single box of shape (N, 4).
        figsize (Tuple[float, float]): size of figure to display.
    Returns:
        None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(boxes, list):
        boxes = [boxes]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, (img, box) in enumerate(zip(imgs, boxes)):
        # img = img.detach()
        # img = TF.to_pil_image(img)
        # box = box.detach()
        img = np.asarray(img)
        if img.shape[0]==3:
            img = np.transpose(img, (1, 2, 0))
        box = np.asarray(box)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].imshow(draw_bounding_boxes(img, box))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None

def show_label_on_img(imgs,labels,figsize=(12.0,12.0)):
    """Displays a single image or list of images with labels.
    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        labels (str): label to be displayed on image.
        figsize (Tuple[float, float]): size of figure to display.
    Returns:
        None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(labels, list):
        labels = [labels]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        # img = img.detach()
        # img = TF.to_pil_image(img)
        img = np.asarray(img)
        if img.shape[0]==3:
            img = np.transpose(img, (1, 2, 0))
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, str(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
        axs[0, i].imshow(np.asarray(img))
    plt.show()

    return None

def model_viewer(model,input_shape = (1,3,128,128), location='model',view=False):
    """
    Function to visualize the model architecture
    Uses torchviz to visualize the model
    Input:
        model: Model to be visualized
        input_shape: Input shape of the model (batch,channels,height,width)
        location: name and location to save the model image. Default is current directory
        view: Boolean to view the model image or not
    Output:
        None
    """
    from torchviz import make_dot
    
    x = torch.randn(input_shape)
    y = model(x)
    dot= make_dot(y.mean(), params=dict(model.named_parameters()),show_attrs=True)
    dot.format = 'png'
    dot.render(location,view=view)
    return None
    
def new_show_cam_on_image(img, mask, use_rgb=True,image_weight: float = 0.4):
    """
    Input:
        img: Image on which the cam is to be superimposed
        mask: cam mask
    Return:
        cam: superimposed image
    """
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # Convert to single channel
    mask = np.float32(mask) / 255 # Convert to float32
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET) # Convert to uint8
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # if np.max(img) > 1:
    #     raise Exception(
    #         "The input image should np.float32 in the range [0, 1]")
    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")
        
    #if use_rgb:
    #    cam = heatmap[..., ::-1] + np.float32(img)
    #else:
    #    cam = heatmap[..., 0] * np.float32(img)
    
    if use_rgb:
        cam = (1 - image_weight) * heatmap[..., ::-1] + image_weight * img
    else:
        cam = (1 - image_weight) * heatmap[..., 0] + image_weight * img
    cam = cam / np.max(cam)
    
    return np.uint8(255 * cam)

def gradcam_viewer(gradcam_layer, model, x_grad,gradcam_rgb=False,use_cuda=False):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    split_val = gradcam_layer.split('.')[1]
    new_layer_name = 'model.' + split_val
    gradcam_eval = eval(new_layer_name)
    with GradCAM(model=model,target_layers=[gradcam_eval],use_cuda=use_cuda) as cm: 
        try:
            if split_val == 'classifier' or 'fc':
                cr = cm(input_tensor=x_grad)[0,:]
        except:
            cr = None
            return cr
        if cr is not None:        
            #cam_img = new_show_cam_on_image(x_grad[0].numpy(),cr,use_rgb=gradcam_rgb)
            x_grad_new = x_grad[0].permute(1,2,0).numpy()
            #cam_img = show_cam_on_image(x_grad_new, cr,use_rgb=gradcam_rgb)
            cam_img = new_show_cam_on_image(x_grad_new, cr,use_rgb=gradcam_rgb)
    return cam_img
    

def plot_classes_pred(images, labels, predictions_prob, preds):
    fig = plt.figure(figsize=(20.0, 60.0))
    if type(preds) == torch.Tensor:
        preds = preds.to('cpu')
    if type(labels) == torch.Tensor:
        labels = labels.to('cpu')
    preds_np = preds.numpy()
    predictions_prob_np = predictions_prob.numpy()
    labels_np = labels.numpy()

    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        images_permuted = images.permute(0, 2, 3, 1)
        plt.imshow(images_permuted[idx,:])
        ax.set_title("{}, {:.1f}%\n(label: {})".format(
                    str(preds_np[idx]),
                    float(predictions_prob_np[idx].max()) * 100.0,
                    str(labels_np[idx])),
                    color=("green" if preds[idx]==labels[idx] else "red"))
    return fig