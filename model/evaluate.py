import json
import argparse

if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--ground_truth_path', default='../dataset/preliminary/ground_truths_example.json', type=str, help='The json file to ground truth')  
    parser.add_argument('--pred_path', default='preds/output.json', type=str, help='The json file to predictions') 

    args = parser.parse_args()  # 解析參數

    with open(args.ground_truth_path, 'r') as file:
      data = json.load(file)
      ground_truth = data['ground_truths']
    
    with open(args.pred_path, 'r') as file:
      data = json.load(file)
      answers = data['answers']

    if (len(ground_truth) != len(answers)):
      raise ValueError("The number of questions in ground truth and prediction files are not same")
    
    n = len(ground_truth)
    c = 0
    nf, ni, nfaq = 0, 0, 0
    cf, ci, cfaq = 0, 0, 0

    for i in range(n):
      s = 0
      if (ground_truth[i]['retrieve'] == answers[i]['retrieve']):
        s = 1
      else:
        print(f"{ground_truth[i]['qid']} - Ans: {ground_truth[i]['retrieve']}, Pred: {answers[i]['retrieve']}")
      c += s
      if (ground_truth[i]['category'] == 'finance'):
        nf += 1
        cf += s
      elif (ground_truth[i]['category'] == 'insurance'):
        ni += 1
        ci += s
      elif (ground_truth[i]['category'] == 'faq'):
        nfaq += 1
        cfaq += s
    
    print(f'Overall Accuracy: {c / n * 100}%')
    print('-' * 50)
    print(f'Finance Accuracy ({cf}/{nf}): {cf / nf * 100}%')
    print(f'Insurance Accuracy ({ci}/{ni}): {ci / ni * 100}%')
    print(f'FAQ Accuracy ({cfaq}/{nfaq}): {cfaq / nfaq * 100}%')
