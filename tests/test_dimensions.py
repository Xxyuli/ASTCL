import configparser
import logging
from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.model import ASTCL, JointAttention, JointConv
from models.layers.GCN import GCN
from models.layers.RevIN import RevIN


torch.set_num_threads(1)


def model(features=1, nodes=5, steps=4):
    return ASTCL([features, 16, 32], [2, 4], 2, 4, features,
                 steps, steps, nodes, dropout=0)


@pytest.mark.parametrize('batch,steps,features', [(2,4,1), (1,4,1), (1,1,1), (2,4,3)])
def test_forward_backward(batch, steps, features):
    m = model(features, steps=steps)
    x = torch.randn(batch, 5, steps, features)
    out = m(x, torch.eye(5).repeat(batch,1,1))
    assert out.shape == (batch, steps)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_attention_token_order():
    m = JointAttention(16, 2, 4)
    # With identity submodules and no shuffle, the ST branch must preserve N,T,C.
    m.T_attention = m.S_attention = m.ST_attention = nn.Identity()
    x = torch.arange(2*16*3*5).reshape(2,16,3,5).float()
    from models.layers.Encoder import channel_shuffle
    expected = channel_shuffle(torch.cat([x[:,:4], x[:,12:], x[:,8:12], x[:,4:8]], 1), 4)
    assert torch.equal(m(x), expected)


def test_jointconv_and_gcn():
    x = torch.randn(2,16,5,4)
    assert JointConv(16,4)(x).shape == x.shape
    g = GCN(16,8,0).eval()
    a = torch.eye(5).repeat(2,1,1)
    assert torch.allclose(g(x,a), g.mlp(torch.cat([x,x,x],dim=1)))


def test_revin_roundtrip():
    x = torch.randn(2,5,4,3)
    r = RevIN(3)
    assert torch.allclose(r(r(x,'norm'),'denorm'), x, atol=1e-6)


def test_invalid_interfaces():
    m = model()
    with pytest.raises(ValueError, match='input must'):
        m(torch.randn(2,4,5,1), torch.eye(5).repeat(2,1,1))
    with pytest.raises(ValueError, match='input must'):
        m(torch.randn(2,5,4,3), torch.eye(5).repeat(2,1,1))
    with pytest.raises(ValueError, match='adj must'):
        m(torch.randn(2,5,4,1), torch.eye(5))
    with pytest.raises(ValueError, match='preserves time'):
        ASTCL([1,16],[2],1,4,1,4,2,5)


@pytest.mark.parametrize('config_path', sorted(Path('config').glob('*.conf')))
def test_shipped_config(config_path):
    c = configparser.ConfigParser(); c.read(config_path); c = c['model']
    m = ASTCL(list(map(int,c['channels'].split(','))), list(map(int,c['num_heads'].split(','))),
              c.getint('depth'), c.getint('partial'), c.getint('num_features'),
              c.getint('t_history'), c.getint('t_pred'), c.getint('num_nodes'))
    assert c.getint('num_features') == 1
    out = m(torch.randn(1,c.getint('num_nodes'),c.getint('t_history'),1),
            torch.eye(c.getint('num_nodes')).unsqueeze(0))
    assert out.shape == (1,c.getint('t_pred'))


def test_training_entry_and_evaluation(tmp_path):
    import train
    import test as evaluation
    from dataloader import preprocess_dataset, CrossDataset
    np.random.seed(7)
    np.save(tmp_path/'adj_mat.npy',np.eye(3))
    np.save(tmp_path/'node_values.npy',np.random.rand(3,50,3).astype('float32')+1)
    paths = preprocess_dataset(str(tmp_path), t_in=2,t_out=2,num_nodes=3,keep_days=0,
                               train=True,val=True,test=True,type='flow')
    loaders = [DataLoader(CrossDataset(*paths[i:i+3]),batch_size=5) for i in (0,3,6)]
    x,y,g = next(iter(loaders[0]))
    assert x.shape[1:] == (3,2,1) and y.shape[1:] == (2,) and g.shape[1:] == (3,3)
    args = Namespace(num_heads='2',channels='1,16',depth=1,partial=4,num_features=1,
        t_history=2,t_pred=2,num_nodes=3,dropout=0,device='cpu',lr=1e-3,
        weight_decay=0,epochs=1,save=True,log_dir=str(tmp_path),log_plotting=str(tmp_path/'plot.txt'))
    _,_,_,m = train.train(args,logging,*loaders[:2])
    train.test(args,logging,m,loaders[2])
    args.warmstart = str(tmp_path/'epoch0_checkpoint.pt')
    evaluation.args = args; evaluation.test_dataloader = loaders[2]
    evaluation.test()
    # Actual train_batch with a singleton final batch must not broadcast its loss.
    loss,out = train.train_batch(m,x[:1],y[:1],g[:1],torch.optim.Adam(m.parameters()),[nn.L1Loss()],'cpu')
    assert out.shape == y[:1].shape and torch.isfinite(loss)
    # Exercise both real CLI entrypoints with generated raw data and a saved checkpoint.
    import shutil
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    (tmp_path/'config').mkdir()
    shutil.copy(root/'config/PEMS08.conf', tmp_path/'config/PEMS08.conf')
    options = ['--device','cpu','--data',str(tmp_path),'--keep_days','0',
               '--num_nodes','3','--num_features','1','--channels','1,16',
               '--num_heads','2','--depth','1','--t_history','2','--t_pred','2',
               '--epochs','1','--batch_size','5']
    result = subprocess.run([sys.executable,str(root/'train.py'),*options],
                            cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = next((tmp_path/'runs').rglob('*_checkpoint.pt'))
    result = subprocess.run([sys.executable,str(root/'test.py'),*options,
                             '--warmstart',str(checkpoint)],
                            cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_output_selects_target_node_and_feature():
    m = ASTCL([3],[],0,4,3,4,4,5,dropout=0).eval()
    with torch.no_grad():
        m.end_conv.weight.zero_(); m.end_conv.bias.zero_()
        for i in range(3):
            m.end_conv.weight[i,i,1,1] = 1
    x = torch.randn(2,5,4,3)
    assert torch.allclose(m(x,torch.eye(5).repeat(2,1,1)), x[:,0,:,0],atol=1e-6)


def test_singleton_evaluation(tmp_path):
    import train
    import test as evaluation
    args = Namespace(num_heads='2',channels='1,16',depth=1,partial=4,num_features=1,
                     t_history=1,t_pred=1,num_nodes=3,dropout=0,device='cpu',
                     warmstart=str(tmp_path/'checkpoint.pt'))
    m = ASTCL([1,16],[2],1,4,1,1,1,3,dropout=0)
    loader = DataLoader(TensorDataset(torch.ones(1,3,1,1),torch.ones(1,1),torch.eye(3).unsqueeze(0)))
    train.test(args,logging,m,loader)
    torch.save({'model_state_dict':m.state_dict()},args.warmstart)
    evaluation.args = args; evaluation.test_dataloader = loader
    evaluation.test()
