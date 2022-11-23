from datasets import Dataset
from setfit import SetFitTrainer, SetFitModel
from sentence_transformers.losses import CosineSimilarityLoss

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def create_hf_dataset(pos_examples:list, neg_examples:list, test_size=None):
    labeled_df = pd.concat([
        DataFrame(dict(text=pos_examples, label=True)), 
        DataFrame(dict(text=neg_examples, label=False))
    ], ignore_index=True)

    labeled_dataset = Dataset.from_pandas(labeled_df)
    if test_size is not None:
        labeled_dataset = labeled_dataset.train_test_split(test_size=test_size)
    
    return labeled_dataset


def train_setfit(model, dataset: Dataset, batch_size=16, num_iterations=20, num_epochs=1, metric="accuracy"):
    if isinstance(model, str):
        model = SetFitModel.from_pretrained(model)

    train_data = dataset['train'] if ('train' in dataset) else dataset
    test_data = dataset['test'] if ('test' in dataset) else None

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        loss_class=CosineSimilarityLoss,
        metric=metric,
        batch_size=batch_size,
        num_iterations=num_iterations,
        num_epochs=num_epochs
    )

    trainer.train()
    if test_data is None:
        return model
    
    metrics = trainer.evaluate()
    return model, metrics


def eval_setfit_loss(model: SetFitModel, eval_data: Dataset):
    pred_diffs = np.abs(model.predict_proba(eval_data['text'])[:,1] - eval_data['label'])
    df = DataFrame(dict(loss=pred_diffs, text=eval_data['text'], label=eval_data['label'])).sort_values('loss', ascending=False)
    return df