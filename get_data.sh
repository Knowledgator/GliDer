img2dataset --url_list data/grit_20m_raw --input_format "parquet"\
    --url_col "url" --caption_col "caption" --output_format webdataset \
    --output_folder data/grit_high_res --processes_count 18 --thread_count 32 --image_size 1080 \
    --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
    --save_additional_columns '["id","noun_chunks","ref_exps","clip_similarity_vitb32","clip_similarity_vitl14"]' \
    --enable_wandb False --retries 0
