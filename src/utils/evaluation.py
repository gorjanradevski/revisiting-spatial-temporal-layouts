class Evaluator:
    def __init__(self, total):
        self.corrects_top1 = 0
        self.corrects_top5 = 0
        self.total = total
        self.best_acc = 0.0

    def reset(self):
        self.corrects_top1 = 0
        self.corrects_top5 = 0

    def process(self, logits, labels):
        self.corrects_top1 += (logits.cpu().argmax(-1) == labels).sum().item()
        self.corrects_top5 += (
            (logits.cpu().topk(k=5).indices == labels.unsqueeze(1)).any(dim=1).sum()
        ).item()

    def evaluate(self):
        top1_accuracy = self.corrects_top1 / self.total
        top5_accuracy = self.corrects_top5 / self.total

        return top1_accuracy, top5_accuracy

    def is_best(self):
        top1_accuracy, top5_accuracy = self.evaluate()
        if (top1_accuracy + top5_accuracy) / 2 > self.best_acc:
            self.best_acc = (top1_accuracy + top5_accuracy) / 2
            return True
        return False
