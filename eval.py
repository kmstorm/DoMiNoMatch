import os
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset

# Add code to draw and save PR curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--alg', type=str, default='fixmatch')
    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=float, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--source_domain', type=str, default='real')
    parser.add_argument('--use_fact_augment', type=bool, default=True)
    parser.add_argument('--train_on_sources_domain', type=bool, default=False)

    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    
    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # specify these arguments manually 
    args.num_labels = 40
    args.ulb_num_labels = 49600
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.num_train_iter = 1024
    dataset_dict = get_dataset(args, args.alg, args.dataset, args.num_labels, args.num_classes, args.data_dir, False)
    eval_dset = dataset_dict['eval']
    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)
 
    acc = 0.0
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []
    results = []  # List to store information for CSV

    with torch.no_grad():
        for data in eval_loader:
            image = data['x_lb']
            target = data['y_lb']

            image = image.type(torch.FloatTensor).cuda()
            feat = net(image, only_feat=True)
            logit = net(feat, only_fc=True)
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1)

            acc += pred.cpu().eq(target).numpy().sum()

            # Collect information for CSV
            for idx, gt, logits in zip(data['idx_lb'], target.cpu().numpy(), logit.cpu().numpy()):
                results.append({
                    "path": eval_dset.data[idx],
                    "ground_truth": gt,
                    "logits": logits.tolist()
                })

            test_feats.append(feat.cpu().numpy())
            test_preds.append(pred.cpu().numpy())
            test_probs.append(prob.cpu().numpy())
            test_labels.append(target.cpu().numpy())

    test_feats = np.concatenate(test_feats)
    test_preds = np.concatenate(test_preds)
    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)

    print(f"Test Accuracy: {acc/len(eval_dset)}")

    wrong_results = [
        {
            "path": result["path"],
            "ground_truth": result["ground_truth"],
            "logits": result["logits"]
        }
        for result, pred in zip(results, test_preds)
        if pred != result["ground_truth"]
    ]
    # Save results to a CSV file
    csv_path = os.path.join(f"results/envmatch/evaluation_results_envmatch_250508_best.csv")
    # os.makedirs(csv_path, exist_ok=True)  # Create the directory if it doesn't exist
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["path", "ground_truth", "logits"])
        writer.writeheader()
        for result in wrong_results:
            writer.writerow(result)

    print(f"Results saved to {csv_path}")

    # Create directory for PR curves if it doesn't exist
    pr_curve_dir = os.path.join("results", "pr_curves")
    os.makedirs(pr_curve_dir, exist_ok=True)

    # Convert labels to one-hot encoding for multi-class PR curve
    y_test_bin = label_binarize(test_labels, classes=range(args.num_classes))

    # Calculate precision-recall curve for each class
    plt.figure(figsize=(10, 8))

    # Colors for different classes
    colors = plt.cm.get_cmap('tab20', args.num_classes)

    # Plot PR curve for each class
    for i in range(args.num_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], test_probs[:, i])
        ap = average_precision_score(y_test_bin[:, i], test_probs[:, i])
        plt.plot(recall, precision, color=colors(i), lw=2,
                label=f'Class {i} (AP = {ap:.2f})')

    # Set plot properties
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curve - {args.dataset} - {args.alg}', fontsize=16)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    pr_curve_path = os.path.join(pr_curve_dir, f"pr_curve_{args.dataset}_{args.alg}.png")
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also create a micro-average PR curve (all classes combined)
    plt.figure(figsize=(10, 8))

    # Calculate micro-average precision and recall
    y_score = test_probs.reshape(-1)
    y_true = y_test_bin.reshape(-1)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score, average="micro")

    plt.plot(recall, precision, color='navy', lw=2,
            label=f'Micro-average PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Micro-average Precision-Recall Curve - {args.dataset} - {args.alg}', fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the micro-average plot
    micro_pr_curve_path = os.path.join(pr_curve_dir, f"micro_pr_curve_{args.dataset}_{args.alg}.png")
    plt.savefig(micro_pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PR curves saved to {pr_curve_dir}")