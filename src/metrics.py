from aggregate_predictions import delete_duplicates_stable

def get_mmr(preds_list, ref):
    # Works properly if has duplicates or n_line_preds < 4

    MMR = 0
    
    for preds, target in zip(preds_list, ref):
        preds = delete_duplicates_stable(preds)
        weights = [1, 0.1, 0.09, 0.08]

        line_MRR = sum(weight * (pred == target)
                       for weight, pred in zip(weights, preds))

        MMR += line_MRR
    
    MMR /= len(preds_list)

    return MMR