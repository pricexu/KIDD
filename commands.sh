# KiDD-D

python GKRRDistill.py --dataset ogbg-molbace --gpc 1 --initialize Random --anm 1 --lrA 1e-6 --lrX 1e-6
python GKRRDistill.py --dataset ogbg-molbace --gpc 10 --initialize Random --anm 2 --lrA 1e-6 --lrX 1e-6
python GKRRDistill.py --dataset ogbg-molbace --gpc 50 --initialize Herding --anm 2 --lrA 1e-6 --lrX 1e-6
python GKRRDistill.py --dataset ogbg-molbbbp --gpc 1 --initialize Herding --anm 2 --lrA 0 --lrX 1e-4
python GKRRDistill.py --dataset ogbg-molbbbp --gpc 10 --initialize Herding --anm 2 --lrA 0 --lrX 1e-4
python GKRRDistill.py --dataset ogbg-molbbbp --gpc 50 --initialize Herding --anm 2 --lrA 0 --lrX 1e-4
python GKRRDistill.py --dataset ogbg-molhiv --gpc 1 --initialize KCenter --anm 4 --lrA 0 --lrX 1e-4
python GKRRDistill.py --dataset ogbg-molhiv --gpc 10 --initialize KCenter --anm 4 --lrA 0 --lrX 1e-2
python GKRRDistill.py --dataset ogbg-molhiv --gpc 50 --initialize KCenter --anm 4 --lrA 0 --lrX 1e-2
python GKRRDistill.py --dataset DD --gpc 1 --initialize Herding --anm 2 --lrA 0 --lrX 1e-4 --batch_size_T 8 --batch_size_S 8 --test_wd 5e-5
python GKRRDistill.py --dataset DD --gpc 10 --initialize Herding --anm 2 --lrA 0 --lrX 1e-4 --batch_size_T 8 --batch_size_S 8 --test_wd 5e-5
python GKRRDistill.py --dataset DD --gpc 50 --initialize Herding --anm 3 --lrA 0 --lrX 1e-4 --batch_size_T 8 --batch_size_S 8 --test_wd 5e-5
python GKRRDistill.py --dataset PROTEINS --gpc 1 --initialize Herding --anm 4 --test_gnn_depth 3  --lrA 0 --lrX 0.001
python GKRRDistill.py --dataset PROTEINS --gpc 10 --initialize Herding --anm 4 --test_gnn_depth 3  --lrA 1e-6 --lrX 0.001
python GKRRDistill.py --dataset PROTEINS --gpc 50 --initialize Herding --anm 4 --test_gnn_depth 2  --lrA 0 --lrX 0.001
python GKRRDistill.py --dataset NCI109 --gpc 1 --initialize Herding --anm 3 --lrA 1e-6 --lrX 0.001
python GKRRDistill.py --dataset NCI109 --gpc 10 --initialize Herding --anm 2 --lrA 1e-6 --lrX 0.001
python GKRRDistill.py --dataset NCI109 --gpc 50 --initialize Herding --anm 2 --lrA 1e-6 --lrX 0.001
python GKRRDistill.py --dataset NCI1 --gpc 1 --initialize Herding --anm 3 --lrA 0 --lrX 0.0001
python GKRRDistill.py --dataset NCI1 --gpc 10 --initialize Herding --anm 3 --lrA 0 --lrX 0.0001
python GKRRDistill.py --dataset NCI1 --gpc 50 --initialize Herding --anm 3 --lrA 0 --lrX 0.0001

# KiDD-LR

python LowRankGKRRDistill.py --dataset ogbg-molbace --gpc 1 --initialize Random --anm 1 --lrA 1e-6 --lrX 1e-6
python LowRankGKRRDistill.py --dataset ogbg-molbace --gpc 10 --initialize Random --anm 2 --lrA 1e-6 --lrX 1e-6
python LowRankGKRRDistill.py --dataset ogbg-molbace --gpc 50 --initialize Herding --anm 2 --lrA 1e-6 --lrX 1e-6 --rank 32 --test_wd 5e-5
python LowRankGKRRDistill.py --dataset ogbg-molbbbp --gpc 1 --initialize Herding --anm 2 --lrA 1e-6 --lrX 1e-3
python LowRankGKRRDistill.py --dataset ogbg-molbbbp --gpc 10 --initialize Herding --anm 3 --lrA 1e-6 --lrX 1e-3 --rank 32
python LowRankGKRRDistill.py --dataset ogbg-molbbbp --gpc 50 --initialize Herding --anm 3 --lrA 1e-6 --lrX 1e-3 --rank 32
python LowRankGKRRDistill.py --dataset ogbg-molhiv --gpc 1 --initialize Herding --anm 4 --lrA 1e-5 --lrX 1e-4
python LowRankGKRRDistill.py --dataset ogbg-molhiv --gpc 10 --initialize Herding --anm 4 --lrA 1e-5 --lrX 1e-2
python LowRankGKRRDistill.py --dataset ogbg-molhiv --gpc 50 --initialize KCenter --anm 4 --lrA 1e-6 --lrX 1e-2 --rank 64
python LowRankGKRRDistill.py --dataset DD --gpc 1 --initialize Herding --anm 2 --lrA 1e-6 --lrX 1e-4 --batch_size_T 8 --batch_size_S 8 --test_wd 5e-5
python LowRankGKRRDistill.py --dataset DD --gpc 10 --initialize Herding --anm 2 --lrA 1e-5 --lrX 1e-4 --batch_size_T 8 --batch_size_S 8 --test_wd 5e-5 --rank 32
python LowRankGKRRDistill.py --dataset DD --gpc 50 --initialize KCenter --anm 3 --lrA 1e-5 --lrX 1e-4 --batch_size_T 8 --batch_size_S 8 --test_wd 5e-5
python LowRankGKRRDistill.py --dataset PROTEINS --gpc 1 --initialize Herding --anm 4 --test_gnn_depth 3  --lrA 1e-5 --lrX 0.001 --rank 32
python LowRankGKRRDistill.py --dataset PROTEINS --gpc 10 --initialize Herding --anm 4 --test_gnn_depth 3  --lrA 1e-6 --lrX 0.001 --rank 16
python LowRankGKRRDistill.py --dataset PROTEINS --gpc 50 --initialize Herding --anm 4 --test_gnn_depth 2  --lrA 1e-5 --lrX 0.001 --rank 32
python LowRankGKRRDistill.py --dataset NCI109 --gpc 1 --initialize Herding --anm 3 --lrA 1e-6 --lrX 0.001
python LowRankGKRRDistill.py --dataset NCI109 --gpc 10 --initialize Herding --anm 2 --lrA 1e-6 --lrX 0.001
python LowRankGKRRDistill.py --dataset NCI109 --gpc 50 --initialize Herding --anm 2 --lrA 1e-6 --lrX 0.001
python LowRankGKRRDistill.py --dataset NCI1 --gpc 1 --initialize Herding --anm 3 --lrA 1e-5 --lrX 0.0001 --rank 24 --test_gnn_depth 3
python LowRankGKRRDistill.py --dataset NCI1 --gpc 10 --initialize Herding --anm 3 --lrA 1e-5 --lrX 0.0001 --rank 24 --test_gnn_depth 4
python LowRankGKRRDistill.py --dataset NCI1 --gpc 50 --initialize Herding --anm 3 --lrA 1e-5 --lrX 0.0001 --rank 24 --test_gnn_depth 5