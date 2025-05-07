from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from core.env_loader import init_env

init_env()

train_dataset = EmbeddingQAFinetuneDataset.from_json("model_api/dataset/finetune_dataset.json")

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-m3",
    model_output_path="output/finetune_bge_m3_model",
    val_dataset=train_dataset
)

finetune_engine.finetune()

embed_model = finetune_engine.get_finetuned_model()
