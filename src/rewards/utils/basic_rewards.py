  class NgramRepeatReward:
    """
    Calculate the number of n-gram repetitions in generated text as reward.
    
    Args:
        ngram_size: Size of n-gram, default is 2.
    """
    
    def __init__(self, ngram_size: int = 2):
        self.ngram_size = ngram_size
    
    def _get_ngram_repeat_nums(self, generated_tokens: list[int]) -> float:
        """
        Calculate the number of n-gram repetitions in generated text.
        
        Args:
            generated_tokens: List of generated tokens.
        
        Returns:
            Number of n-gram repetitions.
        """
        ngrams = []
        for i in range(len(generated_tokens) - self.ngram_size + 1):
            ngram = tuple(generated_tokens[i: i+self.ngram_size])
            ngrams.append(ngram)
        
        counter = Counter(ngrams)
        n_gram_repeat_nums = sum([count-1 for count in counter.values() if count > 1])
        return n_gram_repeat_nums

    def __call__(self, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
        rewards = []
        for i, solution_str in enumerate(solution_strs):
            rewards.append({
                "score": self._get_ngram_repeat_nums(solution_str)
            })
        return rewards