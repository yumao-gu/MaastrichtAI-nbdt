Reference

the original paper and github is: https://github.com/alvinwan/neural-backed-decision-trees.\par
all my experiment results are in 

Pretrained model
We can run the script to get a ./mnist_cnn.pth for the fully connected weights\par

python3 mnist_model_generate.py

Induced Hierarchy
Using the clustering algorithm to generate the decision tree. The tree file is like: ./graph-MNIST-ave-l1.json and the html file in ./out. We can choose the distance metrice and linkage method as follow:\par

python3 mnist_hierarchy.py --induced-linkage average --induced-affinity l1

Fine-tuning
We can fine-tune our networks using hard/soft loss then. The results will be saved in ./checkpoint\par

python3 mnist_main.py --resume --path-graph=./graph-MNIST-min-l2.json --loss=HardTreeSupLoss --analysis=HardEmbeddedDecisionRules --tree-supervision-weight 1 --epochs 401 --batch-size=512

Analysis
You can use this line to test your new networks and the results and log file will be save in ./data/MNIST/{data-folder}\par

python3 mnist_test_net.py --loss=HardTreeSupLoss --pretrained --path-resume ./checkpoint/ckpt-MNIST-ResNet18-MNIST-min-l2-HardTreeSupLoss-wtl-best.pth --path-graph ./graph-MNIST-min-l2.json --analysis=HardEmbeddedDecisionRules --data-folder wn-min-l2-weighted-hard-max

Trials and Improvements
I did some experiments to improve the current accuracy. You can compare with the report and find the corresponding codes.\par
4.1 mnist_model_generate.py
4.2 mnist_hierarchy.py
4.3 model.py/EmbeddedDecisionRules/get_node_logits()
4.4 loss.py/Hard(Soft)TreeSupLoss/forward()
4.5 mnist_model_generate.py/line20~21

