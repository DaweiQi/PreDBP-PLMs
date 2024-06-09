from transformers import T5EncoderModel, T5Tokenizer
import torch
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))


def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    return model, tokenizer


def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=3000, max_batch=1):
    res_names = []
    if sec_struct:
        pass
    results = {"residue_embs": dict(),
               "protein_embs": dict(),
               "sec_structs": dict()
               }
    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))
        # count residues in current batch and add the last seq length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            print(pdb_ids)
            res_names.append(pdb_ids)
            print(seq_lens)
            batch = list()
            # add_special_tokens adds extra token at the end of each seq
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            # if sec_struct:  # in case you want to predict secondary structure from embeddings
            #     d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if per_residue:  # store per-residue embeddings (Lx1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    return results, res_names


def get_seq(path, filename):
    from Bio.SeqIO import parse
    path = path + filename
    file = open(path)
    # Parse the content of the sequence file and return it as a list of SeqRecord objects
    records = parse(file, "fasta")
    # Use Python for looping through records and printing the properties of the sequence record (SqlRecord),
    # such as ID, name, description, sequence data, etc
    seqs = []
    names = []
    cnt = 0
    greater_512 = 0
    for record in records:
        names.append(str(record.name) + str(cnt))
        tmp = str(record.seq)
        len(tmp)
        seqs.append(tmp)
        cnt = cnt + 1
    seq_dict = dict(zip(names, seqs))
    print(greater_512)
    print(len(seq_dict))
    return seq_dict


if __name__ == '__main__':
    model, tokenizer = get_T5_model()
    import gc

    gc.collect()
    per_residue = 0
    per_protein = 1
    sec_struct = 0

    path = '../Data/DBPs/PDB1075/'
    filename = 'PDB1075_train_P.txt'
    seq_dict = get_seq(path, filename)
    results, res_names = get_embeddings(model, tokenizer, seq_dict,
                                        per_residue, per_protein, sec_struct)
    import numpy as np

    res_names = np.array(res_names, dtype=object)
    print(res_names[0])
    print(res_names)
    print(results["protein_embs"])
    print(len(results["protein_embs"]))

    import pickle
    import numpy as np

    pickle_filepath = "../Pre-trained_features/DBPs/" + filename + '_t5.pkl'
    with open(pickle_filepath, "wb") as tf:
        pickle.dump(results["protein_embs"], tf)

    # # Read pickle file
    # with open(pickle_filepath, "rb") as tf:
    #     feature_dict = pickle.load(tf)
    # feature_N = np.array([item for item in feature_dict.values()])
    # print(feature_N.shape)
    #
    # # Save as npz file
    # npz_filepath = "../Features/Pre_trained_features/" + filename + '_t5.npz'
    # np.savez(npz_filepath, feature_N)
