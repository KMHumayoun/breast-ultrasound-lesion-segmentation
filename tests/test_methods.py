import numpy as np
import pytest
from study.clinical import audit_groups,segmentation_metrics,patient_ci,binary_metrics

def test_patient_overlap_is_rejected():
    records=[{'case_id':'a','patient_id':'p','split':'train'},{'case_id':'b','patient_id':'p','split':'test'}]
    with pytest.raises(ValueError):audit_groups(records)

def test_empty_masks_and_spacing_semantics():
    a=np.zeros((8,8),bool);assert segmentation_metrics(a,a,[1,1])['dice']==1
    b=a.copy();b[2:4,2:4]=1;r=segmentation_metrics(a,b,[2,3]);assert r['dice']==0 and np.isinf(r['hd95_spacing_units']) and r['absolute_volume_error']==24
    with pytest.raises(ValueError):segmentation_metrics(a,b,[0,1])

def test_uncertain_labels_are_excluded_and_ci_resamples_patients():
    assert binary_metrics([0,1,np.nan],[.1,.9,.5])['known_labels']==2
    result=patient_ci([0.,0.,1.,1.],['a','a','b','b']);assert result['patients']==2 and result['patient_mean']==.5

from study.experiment import preprocess,features

def test_resizing_retains_binary_mask_and_converts_physical_units():
    image=np.ones((40,80))*.5;mask=np.zeros((40,80),bool);mask[10:30,20:60]=1
    a,m,spacing=preprocess(image,mask,.01,20)
    assert a.shape==m.shape==(20,20) and m.dtype==bool
    np.testing.assert_allclose(spacing,[.2,.4])

def test_local_features_have_no_reference_mask_input():
    a=np.ones((16,16),np.float32)*.5;f=features(a)
    assert f.shape==(256,6) and np.isfinite(f).all()


def test_experiment_summary_json_accepts_scientific_counts_and_intervals():
    import json
    from study.experiment import scientific_json
    payload={'empty_output_count':np.sum(np.array([True,False])), 'interval':np.array([.2,.8])}
    restored=json.loads(json.dumps(payload,default=scientific_json))
    assert restored=={'empty_output_count':1,'interval':[.2,.8]}
