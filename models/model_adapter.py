from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def get_bag_prediction(self, patches):
        pass

    @abstractmethod
    def get_instance_weights(self, patches, threshold=None):
        pass

    @abstractmethod
    def get_count_prediction(self, patches, threshold=None):
        pass

# This adapter allows us to use the same interface for both Attention and GatedAttention models,
# making it easier to switch between them in our training and evaluation code without needing to 
# change the logic for extracting predictions, attention weights, or count predictions.

class AttentionModelAdapter(ModelAdapter):
    def get_bag_prediction(self, patches):
        Y_prob, predicted_label, _ = self.model(patches)
        return Y_prob, predicted_label

    def get_instance_weights(self, patches, threshold=None):
        _, _, attention_weights = self.model(patches)
        return attention_weights

    def get_count_prediction(self, patches, threshold=None):
        _, _, A = self.forward(patches)
        K = A.shape[1]  # number of instances
        if threshold is None:
            threshold = 1.0 / K
        count = int((A.squeeze(0) > threshold).sum().item())
        return count



