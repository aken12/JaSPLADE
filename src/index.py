from os.path import join as oj
from tqdm import tqdm
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader

from indexing.index_array import IndexDictOfArray
from dataset import EncodeDataset
from collator import EncodeCollator
from models.splade import Splade
from argument import get_index_args

def indexing(args):
    model = Splade(model_type_or_dir=args.model_name)
    tokenizer = model.transformer_rep.tokenizer
    model.to(args.device)

    encode_collator = EncodeCollator(data_args=args,tokenizer=tokenizer)

    if args.collection_path != None:
        args.dataset_name = args.collection_path
    encode_dataset = EncodeDataset(data_args=args)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=args.batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
    )

    dim_voc = model.transformer_rep.transformer.config.vocab_size
    sparse_index = IndexDictOfArray(args.index_output_path, dim_voc=dim_voc, force_new=True)    
    count = 0
    doc_ids = []
    logger.info("index process started...")
    with torch.no_grad():
        model.eval()
        for batch in tqdm(encode_loader, desc="Indexing", position=0, leave=True):
            inputs = {k: v.to(args.device) for k, v in batch[1].items()}
            batch_documents = model(d_kwargs=inputs)["d_rep"]
            
            row, col = torch.nonzero(batch_documents, as_tuple=True)
            data = batch_documents[row, col]
            row = row + count

            batch_ids = list(batch[0])
            doc_ids.extend(batch_ids)
            count += len(batch_ids)
            sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                               n_docs=len(batch_ids))

    sparse_index.save()
    pickle.dump(doc_ids, open(oj(args.index_output_path, "doc_ids.pkl"), "wb"))
    logger.info("Done iterating over the corpus!")
    logger.info("index contains {} posting lists".format(len(sparse_index)))
    logger.info("index contains {} documents".format(len(doc_ids)))

def main():
    args = get_index_args()
    indexing(args)

if __name__=="__main__":
    main()