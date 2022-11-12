# Environment Settings

# Data
## Cora
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

|  | Task | # Graphs | # Nodes (Avg) | # Edges (Avg) | # Features | # Classes |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | Node classification | 1 | 2708 | 5429 | 1433 | 7 |
| Protein | Graph classification | 1113 | 39.06  | 72.81 | 4 | 2 |



* train 140 [0:140]
* val 500 [140:640]
* test 1000 [-1000:]

1640 -> 984 : 328 : 328

# GNN Training
## hyperparameter








# Directory
Eventually, the directory structure should look like this:

* this repo
  * Genetic-GNN
  * README.md

# Question
1. edge_index가 2개 edge간에 관계인가?
2. nodes 2708인데 1640개만 쓰는 이유?
    * 답
3. test가 epoch마다 돌아가는지 확인, 왜 그렇게 쓰는지