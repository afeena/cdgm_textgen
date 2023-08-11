import argparse

import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--dictionary")
    parser.add_argument("--output")
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--mbart", action="store_true")

    args = parser.parse_args()

    model = torch.load(args.model_path)
    print(model.keys())

    if args.last:
        embed = torch.transpose(model["model"]["decoder.output_projection.weight"], 0, 1)
    else:
        embed = model["model"]["decoder.embed_tokens.weight"]

    # save embedding matrix alone
    torch.save(embed, f'{args.output}.pt')

    # possibly save memory
    n_words = embed.shape[0]

    if args.mbart:
        special_tokens = ["<bos>", "<pad>", "<eos>", "<unk>"]
    else:
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]

    dictionary = {v: i for i, v in enumerate(special_tokens)}

    with open(args.dictionary, 'r', encoding="utf-8") as d:
        lines = d.readlines()
        for line in lines:
            token, field = line.rstrip().rsplit(" ", 1)
            if token in special_tokens:
                continue
            dictionary[token] = len(dictionary)

    if args.mbart:
        # mbart vocab specific
        # langs = model["args"].langs.split(",")
        langs = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT",
                 "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK",
                 "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE",
                 "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN",
                 "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]
        for l in langs:
            if l in special_tokens:
                continue
            dictionary["[{}]".format(l)] = len(dictionary)
        dictionary["<mask>"] = len(dictionary)

        assert len(dictionary) == n_words
        dictionary_inv = {ind: tok for tok, ind in dictionary.items()}

        with open(f'{args.output}.vocab.txt', 'w') as out:
            for k in range(len(dictionary)):
                if dictionary_inv[k] in special_tokens:
                    continue
                print(dictionary_inv[k], file=out)

    # cdgm_textgen reuires this format as an input for pre-trained embeddings
    with open(f"{args.output}.txt", "w") as out:
        out.write("{} {}".format(embed.shape[0], embed.shape[1]))
        out.write("\n")

        for tok, ind in dictionary.items():
            vals = " ".join([str(x) for x in embed[ind, :].cpu().data.numpy()])
            out.write(f"{tok} {vals}\n")
