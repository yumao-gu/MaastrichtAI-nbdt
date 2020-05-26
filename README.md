**Reference**

All work is in /assignment folder.

The original paper and github is: https://github.com/alvinwan/neural-backed-decision-trees.

All my experiment results are in https://drive.google.com/drive/folders/1X9jjfJAHhc2uWY7IJzIw8jhgm-DF_iwH

**Pretrained model**

We can run the script to get a ./mnist_cnn.pth for the fully connected weights.

python3 mnist_model_generate.py

**Induced Hierarchy**

Using the clustering algorithm to generate the decision tree. The tree file is like: ./graph-MNIST-ave-l1.json and the html file in ./out. We can choose the distance metrice and linkage method as follow.

python3 mnist_hierarchy.py --induced-linkage average --induced-affinity l1

**Fine-tuning**

We can fine-tune our networks using hard/soft loss then. The results will be saved in ./checkpoint.

python3 mnist_main.py --resume --path-graph=./graph-MNIST-min-l2.json --loss=HardTreeSupLoss --analysis=HardEmbeddedDecisionRules --tree-supervision-weight 1 --epochs 401 --batch-size=512

**Analysis**

You can use this line to test your new networks and the results and log file will be save in ./data/MNIST/{data-folder}.

python3 mnist_test_net.py --loss=HardTreeSupLoss --pretrained --path-resume ./checkpoint/ckpt-MNIST-ResNet18-MNIST-min-l2-HardTreeSupLoss-wtl-best.pth --path-graph ./graph-MNIST-min-l2.json --analysis=HardEmbeddedDecisionRules --data-folder wn-min-l2-weighted-hard-max

**Trials and Improvements**

I did some experiments to improve the current accuracy. You can compare with the report and find the corresponding codes.

4.1 mnist_model_generate.py

4.2 mnist_hierarchy.py

4.3 model.py/EmbeddedDecisionRules/get_node_logits()

4.4 loss.py/Hard(Soft)TreeSupLoss/forward()

4.5 mnist_model_generate.py/line20~21

