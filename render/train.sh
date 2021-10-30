python main.py --name render_lrw_lr_2e-4_vggsingle --lr 2e-4 --it 1000000 --batch 6 -tl 3
python main.py --name render_lrw_lr_2e-4_blend_2 --lr 2e-4 --it 1000000 --batch 8
python main.py --name tune --lr 1e-5 --it 2100 --batch 2 -tl 1
python tune.py --lr 1e-5 --it 5000 --batch 2 -tl 1
python tune.py --lr 1e-4 --it 10000 --batch 8 -tl 1
python tune_test.py --lr 1e-5 --it 2100 --batch 2 -tl 1
python nvp_test.py --lr 1e-5 --it 2100 --batch 2 -tl 1
python change_tex.py --lr 1e-5 --it 2100 --batch 2 -tl 1