import numpy as np
import shap
import pickle
from datasets import load_dataset
from src.utils import legacy_get_dataset_info
from transformers import pipeline, AutoTokenizer, TFBertModel
import pandas as pd
from datasets import load_dataset, Dataset
import os
from tqdm import tqdm
from src.utils import token_segments, text_ft_index_ends, format_text_pred, ConfigLoader
from sklearn.model_selection import train_test_split

from transformers import BertConfig

from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFAutoModel
import tensorflow as tf


# from src.models import Model
from src.models import AllAsTextModel
from src.joint_masker import JointMasker
import scipy as sp

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--ds_type",
#     type=str,
#     default="airbnb",
#     help="Name of dataset to use",
# )
# parser.add_argument(
#     "--text_model_code",
#     type=str,
#     default="disbert",
#     help="Code name for text model to use",
# )
# parser.add_argument(
#     "--repeat_idx",
#     type=int,
#     default=None,
#     help="For the explainability consistency experiment, which repeat to use",
# )

def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred) 
    f1 = 2 * tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-16)
    return f1


def run_shap(merged_data, max_samples=100, test_set_size=100):
    # # Shap args
    # args = ConfigLoader(config_type, "configs/shap_configs.yaml")
    # # Dataset info
    # di = ConfigLoader(args.dataset, "configs/dataset_configs.yaml")
    # Data
    all_text_versions = ["all_as_text"]
    # ds_name = (
    #     di.all_text_dataset
    #     if args["version"] in all_text_versions
    #     else di.ordinal_dataset
    # )
    # train_df = load_dataset(
    #     ds_name,
    #     split="train",  # download_mode="force_redownload"
    # ).to_pandas()
    # y_train = train_df[di.label_col]
    # test_df = load_dataset(
    #     ds_name,
    #     split="test",  # download_mode="force_redownload"
    # ).to_pandas()

    train_df, test_df = train_test_split(merged_data, random_state = 42, test_size = 0.25)
    y_train = train_df['AdoptionSpeed']
    test_df = test_df.sample(test_set_size, random_state=11)

    model1 = load_model("/cephfs/DSC261/project/model/best_bert_unimodal.hdf5", custom_objects={'f1_metric': f1_metric, 'TFBertModel': TFBertModel})
    config = BertConfig.from_pretrained('bert-base-uncased')  
    model1.config = config

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, max_length = 256,
                                             padding = 'max_length', truncation = True, 
                                              return_token_type_ids= False, return_tensors = 'tf')
    text_pipeline = pipeline(
        "text-classification",
        model=model1,
        tokenizer=tokenizer,
        truncation=True,
        padding=True,
        top_k=None,
    )
        # Define how to convert all columns to a single string
    cols_to_str_fn = lambda array: " | ".join(
        [f"{col} {str(val)}" for col, val in zip(train_df[['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'Description']].columns, array)])

    model = AllAsTextModel(
        text_pipeline=text_pipeline,
        cols_to_str_fn=cols_to_str_fn,
    )

    np.random.seed(11)
    x = test_df[['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'Description']].values

    # We need to load the ordinal dataset so that we can calculate the correlations for
    # the masker
    ord_train_df = train_df.copy()

    tab_pt = sp.cluster.hierarchy.complete(
        sp.spatial.distance.pdist(
            ord_train_df[['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']]
            .values.T,
            metric="correlation",
        )
    )

    masker = JointMasker(
        tab_df=train_df[['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']],
        text_cols=train_df['Description'],
        cols_to_str_fn=cols_to_str_fn,
        tokenizer=tokenizer,
        collapse_mask_token=True,
        max_samples=max_samples,
        tab_partition_tree=tab_pt,
    )

    
    
    explainer = shap.explainers.Partition(model=model.predict, masker=masker)
    shap_vals = explainer(x)

    output_dir = "./model/shap_vals/"
    print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"pet_bert_alltext_shap.pkl"), "wb") as f:
        pickle.dump(shap_vals, f)

    return shap_vals


def run_all_text_baseline_shap(
    merged_data,
    # config_type,
    test_set_size=100,
):
    train_df, test_df = train_test_split(merged_data, random_state = 42, test_size = 0.25)
    y_train = train_df['AdoptionSpeed']
    test_df = test_df.sample(test_set_size, random_state=11)
    
    # Shap args
    # args = ConfigLoader(config_type, "configs/shap_configs.yaml")
    # # Dataset info
    # di = ConfigLoader(args.dataset, "configs/dataset_configs.yaml")
    # Data
    # test_df = load_dataset(
    #     di.ds_name, split="test", download_mode="force_redownload"
    # ).to_pandas()
    # test_df = test_df.sample(test_set_size, random_state=55)

    # Models

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, max_length = 256,
                                             padding = 'max_length', truncation = True, 
                                              return_token_type_ids= False, return_tensors = 'tf')
    text_pipeline = pipeline(
        "text-classification",
        model=model1,
        tokenizer=tokenizer,
        truncation=True,
        padding=True,
        top_k=None,
    )
    
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.text_model_base, model_max_length=512
    # )
    # text_pipeline = pipeline(
    #     "text-classification",
    #     model=args.my_text_model,
    #     tokenizer=tokenizer,
    #     device="cuda:0",
    #     truncation=True,
    #     padding=True,
    #     top_k=None,
    # )

    # Define how to convert all columns to a single string
    def cols_to_str_fn(array):
        return " | ".join(
            [
                f"{col} {val}"
                for col, val in zip(
                    train_df.drop('AdoptionSpeed', axis = 1).columns, array
                )
            ]
        )

    np.random.seed(1)
    x = list(
        map(
            cols_to_str_fn,
            test_df.drop('AdoptionSpeed', axis = 1).values,
        )
    )
    explainer = shap.Explainer(text_pipeline, tokenizer)
    shap_vals = explainer(x)

    output_dir = "./models/shap_vals/"
    print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"pet_bert_alltext_shap_baseline.pkl"), "wb") as f:
        pickle.dump(shap_vals, f)

    return shap_vals


# def load_shap_vals(config_name, add_parent_dir=False):
#     pre = "../" if add_parent_dir else ""  # for running from notebooks
#     with open(f"{pre}models/shap_vals/{config_name}.pkl", "rb") as f:
#         shap_vals = pickle.load(f)
#     return shap_vals


def gen_summary_shap_vals(merged_data, ctype = 'pet_bert_alltext_shap',add_parent_dir=False): #config_type
    # Shap args fpath = "./models/shap_vals/pet_bert_alltext_shap.pkl"
    # args = ConfigLoader(config_type, "configs/shap_configs.yaml")
    # Dataset info
    # di = ConfigLoader(args.dataset, "configs/dataset_configs.yaml")
    
    # shap_vals = load_shap_vals(config_type, add_parent_dir=add_parent_dir)
    with open(f"./models/shap_vals/{ctype}.pkl", "rb") as f: #f"./models/shap_vals/pet_bert_alltext_shap.pkl"
        shap_vals = pickle.load(f)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, max_length = 256,
                                             padding = 'max_length', truncation = True, 
                                              return_token_type_ids= False, return_tensors = 'tf')

    filepath = f"./models/shap_vals/summed_{ctype}.pkl"
    print(
        f"""
            #################
            {ctype}
            #################
            """
    )
    if "baseline" not in ctype:
        grouped_shap_vals = []
        for label in range(2): #no. of labels
            shap_for_label = []
            for idx in tqdm(range(len(shap_vals))):
                sv = shap_vals[idx, :, label]
                text_ft_ends = text_ft_index_ends(
                    sv.data[-1:], tokenizer
                )
                text_ft_ends = [len(merged_data.columns)-2] + [
                    x + len(merged_data.columns)-2 + 1
                    for x in text_ft_ends
                ]
                val = np.append(
                    sv.values[: -1],
                    [
                        np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                    + [np.sum(sv.values[text_ft_ends[-1] :])],
                )

                shap_for_label.append(val)
            grouped_shap_vals.append(np.vstack(shap_for_label))
        print(f"Saving to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(np.array(grouped_shap_vals), f)

    else:
        col_name_filepath = f"./models/shap_vals/summed_{ctype}_col_names.pkl"
        colon_filepath = f"./models/shap_vals/summed_{ctype}_colons.pkl"
        grouped_shap_vals = []
        grouped_col_name_shap_vals = []
        grouped_colon_shap_vals = []
        for label in range(2):
            shap_for_label = []
            shap_for_col_name = []
            shap_for_colon = []
            for idx in tqdm(range(len(shap_vals))):
                sv = shap_vals[idx, :, label]
                stripped_data = np.array([item.strip() for item in sv.data])
                text_ft_ends = (
                    [1] + list(np.where(stripped_data == "|")[0]) + [len(sv.data) + 1]
                )
                # Need this if there are | in the text that aren't col separators
                # Not super robust and only implemented for the current col to text
                # mapping, but works for now
                if (
                    len(text_ft_ends) != len(merged_data.columns)
                ):
                    text_ft_ends = (
                        [1]
                        + [
                            i
                            for i in list(np.where(stripped_data == "|")[0])
                            if sv.data[i + 1].strip()
                            in [
                                token_segments(col, tokenizer)[0][1].strip()
                                for col in merged_data.drop('AdoptionSpeed', axis = 1).columns
                            ]
                            + merged_data.drop('AdoptionSpeed', axis = 1).columns
                            # + di.categorical_cols
                            # + di.numerical_cols
                            # + di.text_cols
                        ]
                        + [len(sv.data) + 1]
                    )
                assert (
                    len(text_ft_ends)
                    == len(merged_data.drop('AdoptionSpeed', axis = 1).columns) + 1
                )
                val = np.array(
                    [
                        np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                )
                colon_idxs = np.where(stripped_data == ":")[0]
                col_idxs_after_ft = [
                    colon_idxs[list(np.where(colon_idxs > te)[0])[0]]
                    for te in text_ft_ends[:-1]
                ]
                ft_name_vals = np.array(
                    [
                        np.sum(sv.values[text_ft_ends[i] : col_idxs_after_ft[i]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                )
                colon_vals = np.array(sv.values[col_idxs_after_ft])
                shap_for_label.append(val)
                shap_for_col_name.append(ft_name_vals)
                shap_for_colon.append(colon_vals)
            grouped_shap_vals.append(np.vstack(shap_for_label))
            grouped_col_name_shap_vals.append(shap_for_col_name)
            grouped_colon_shap_vals.append(shap_for_colon)
        print(f"Saving to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(np.array(grouped_shap_vals), f)
        print(f"Saving to {col_name_filepath}")
        with open(col_name_filepath, "wb") as f:
            pickle.dump(np.array(grouped_col_name_shap_vals), f)
        print(f"Saving to {colon_filepath}")
        with open(colon_filepath, "wb") as f:
            pickle.dump(np.array(grouped_colon_shap_vals), f)


# def load_shap_vals_legacy(
#     ds_name,
#     text_model_code,
#     add_parent_dir=True,
#     tab_scale_factor=2,
#     repeat_idx=None,
# ):
#     # pre = "../" if add_parent_dir else ""  # for running from notebooks
#     tab_pre = f"_sf{tab_scale_factor}" if tab_scale_factor != 2 else ""
#     repeat_idx_str = f"_{repeat_idx}" if repeat_idx is not None else ""
#     text_model_name = f"_{text_model_code}"
    

#     with open(
#         f"./models/shap_vals{text_model_name}{tab_pre}{repeat_idx_str}/{ds_name}/shap_vals_all_text.pkl",
#         "rb",
#     ) as f:
#         shap_all_text = pickle.load(f)
#     with open(
#         f"{pre}models/shap_vals{text_model_name}{tab_pre}{repeat_idx_str}/{ds_name}/shap_vals_all_text_baseline.pkl",
#         "rb",
#     ) as f:
#         shap_all_text_baseline = pickle.load(f)
#     return (
#         [shap_all_text, shap_all_text_baseline],
#         [
#             "all_text",
#             "all_text_baseline",
#         ],
#     )


if __name__ == "__main__":
    config_type = 'pet_bert_alltext_shap' #parser.parse_args().config
    if "baseline" in config_type:
        run_all_text_baseline_shap(merged_data, test_set_size=1000)

    else:
        run_shap(merged_data, test_set_size=1000)
    gen_summary_shap_vals(merged_data, ctype = config_type)
