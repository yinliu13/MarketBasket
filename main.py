from data import BasketConstructor
from knndtw import KnnDtw
from embedding_wrapper import EmbeddingWrapper
from helper import nested_change, remove_products_which_are_uncommon, remove_short_baskets, split_data
from sklearn.model_selection import train_test_split

def run():
    embedding_wrapper = EmbeddingWrapper('product')
    bc = BasketConstructor('./data/', './data/')
    ub_basket = bc.get_baskets('prior', reconstruct=False)
    ok, ub_basket = train_test_split(ub_basket, test_size=0.1, random_state=0)
    #embedding_wrapper = EmbeddingWrapper('tafeng_products')
    print(ub_basket)

    all_baskets = ub_basket.basket.values
    print(all_baskets)
    #changes every item to string
    print("nested change")
    all_baskets = nested_change(list(all_baskets), str)
    print("embedding_wrapper.remove_products_wo_embeddings(all_baskets)")
    all_baskets = embedding_wrapper.remove_products_wo_embeddings(all_baskets)
    print("uncommon products")
    all_baskets = remove_products_which_are_uncommon(all_baskets)
    print("short baskets")
    all_baskets = remove_short_baskets(all_baskets)
    print("nested change")

    all_baskets = nested_change(all_baskets, embedding_wrapper.lookup_ind_f)
    print("split_data")
    train_ub, val_ub_input, val_ub_target, test_ub_input, test_ub_target = split_data(all_baskets)
    print('knndtw')
    knndtw = KnnDtw(n_neighbors=[5])
    preds_all, distances = knndtw.predict(train_ub, val_ub_input, embedding_wrapper.basket_dist_EMD, 
                                          embedding_wrapper.basket_dist_REMD)
    print(preds_all)
    print(distances)
    #print("Wasserstein distance", sum(distances)/len(distances))
    return preds_all, distances


if __name__ == "__main__":
    run()
