import numpy as np
import pytest
torch=pytest.importorskip('torch')
from study.networks import CompactUNet,BinaryDeepLab,segmentation_loss
from study.neural import flip_pair


def test_unet_shape_and_finite_backward():
    model=CompactUNet();x=torch.rand(2,1,32,32);mask=(x>.5).float();out=model(x)
    assert out.shape==mask.shape
    loss=segmentation_loss(out,mask);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_deeplab_binary_head_without_weight_download():
    model=BinaryDeepLab(False).eval()
    with torch.no_grad():out=model(torch.rand(1,1,32,32))
    assert out.shape==(1,1,32,32) and torch.isfinite(out).all()


def test_flip_preserves_image_mask_correspondence():
    mask=np.zeros((8,8),bool);mask[2:5,1:3]=1;image=mask.astype(float)
    x,y=flip_pair(image,mask,True);np.testing.assert_array_equal(x>0,y)
    np.testing.assert_array_equal(x,np.fliplr(image))
