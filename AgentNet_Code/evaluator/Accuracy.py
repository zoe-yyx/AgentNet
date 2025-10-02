class Accuracy:
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0

    def update(self, is_correct):
        self.total_count += 1
        if is_correct:
            self.correct_count += 1

    def get_accuracy(self):
        return self.correct_count / self.total_count if self.total_count > 0 else 0

