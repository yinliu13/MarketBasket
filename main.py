from data import BasketConstructor
from knndtw import KnnDtw
from embedding_wrapper import EmbeddingWrapper
from helper import nested_change, remove_products_which_are_uncommon, remove_short_baskets, split_data
from sklearn.model_selection import train_test_split
import numpy as np
def run():
    embedding_wrapper = EmbeddingWrapper('product')
    bc = BasketConstructor('./data/', './data/')
    ub_basket = bc.get_baskets('prior', reconstruct=False)
    ok, ub_basket = train_test_split(ub_basket, test_size=0.20, random_state=0)
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
    medium_baskets, all_baskets = remove_short_baskets(all_baskets)
    print(medium_baskets , all_baskets)
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

def runtopncustomers():
    embedding_wrapper = EmbeddingWrapper('product')
    bc = BasketConstructor('./data/', './data/')
    ub_basket = bc.get_baskets('prior', reconstruct=False)
    ok, ub_basket = train_test_split(ub_basket, test_size=0.20, random_state=0)
    #embedding_wrapper = EmbeddingWrapper('tafeng_products')
    #print(ub_basket)

    all_baskets = ub_basket.basket.values
    #print(all_baskets)
    #changes every item to string
    print("nested change")
    all_baskets = nested_change(list(all_baskets), str)
    print("embedding_wrapper.remove_products_wo_embeddings(all_baskets)")
    all_baskets = embedding_wrapper.remove_products_wo_embeddings(all_baskets)
    #print('test' ,all_baskets)
    #every customer sequence
    for s in range(2):
        print(all_baskets[s])
        #itemperklant = np.array([])
        itemperklant = []
        sizes = []
        top_nc = get_top_nc(all_baskets, 2)
        for i in range(len(all_baskets[s])):   # every basket in all baskets
            for j in range(len(all_baskets[s][i])): # every item in every basket
                #print('basket', all_baskets[s][i][j])
                itemperklant.append( all_baskets[s][i][j])
        print(itemperklant)
        unique_items = np.unique(itemperklant)
        print(unique_items)
        arrayklant = np.zeros((int(len(unique_items)), 2))
        arrayklant[:, 0] = unique_items
        for ding in range(len(unique_items)):
            countproduct = itemperklant.count(unique_items[ding])
            # itemperklant.append(countproduct)
            arrayklant[ding, 1] = countproduct

        print(arrayklant)

        sorted = arrayklant[np.argsort(arrayklant[:, 1])]
        print('sorted', sorted)
        product = np.array([])
        print('average length', top_nc[s])
        for reverse in range(int(top_nc[s])):
            print   ('test', sorted[-reverse - 1, :])
            product = np.append(product, sorted[-reverse, :])


    #print("uncommon products")
    #all_baskets = remove_products_which_are_uncommon(all_baskets)
    #print("short baskets")
    #medium_baskets, all_baskets = remove_short_baskets(all_baskets)
    #print(medium_baskets , all_baskets)
    #print("nested change")
    #all_baskets = nested_change(all_baskets, embedding_wrapper.lookup_ind_f)
    #print("split_data")
    train_ub, val_ub_input, val_ub_target, test_ub_input, test_ub_target = split_data(all_baskets)
    #print('knndtw')
    #knndtw = KnnDtw(n_neighbors=[5])
    #preds_all, distances = knndtw.predict(train_ub, val_ub_input, embedding_wrapper.basket_dist_EMD, embedding_wrapper.basket_dist_REMD)
    #print(preds_all)
    #print(distances)
    #print("Wasserstein distance", sum(distances)/len(distances))
    #return preds_all, distances
    write_path = 'data/testprint'
    with open(write_path + '.txt', 'w') as results:
        results.write('All baskets test ' + str(all_baskets) + '\n')
    results.close()
def runtopnglobal():
    embedding_wrapper = EmbeddingWrapper('product')
    bc = BasketConstructor('./data/', './data/')
    ub_basket = bc.get_baskets('prior', reconstruct=False)
    ok, ub_basket = train_test_split(ub_basket, test_size=0.20, random_state=0)
    #embedding_wrapper = EmbeddingWrapper('tafeng_products')
    #print(ub_basket)

    all_baskets = ub_basket.basket.values
    #print(all_baskets)
    #changes every item to string
    print("nested change")
    all_baskets = nested_change(list(all_baskets), str)
    print("embedding_wrapper.remove_products_wo_embeddings(all_baskets)")
    all_baskets = embedding_wrapper.remove_products_wo_embeddings(all_baskets)
    #print('test' ,all_baskets)
    #every customer sequence
    itemsalleklanten = []
    for s in range(2):
        print(all_baskets[s])

        sizes = []
        top_nc = get_top_nc(all_baskets, 2)
        for i in range(len(all_baskets[s])):   # every basket in all baskets

            for j in range(len(all_baskets[s][i])): # every item in every basket
                #print('basket', all_baskets[s][i][j])
                itemsalleklanten.append(all_baskets[s][i][j])
        print(itemsalleklanten)
        unique_items = np.unique(itemsalleklanten)
        print(unique_items)
        arrayklant = np.zeros((int(len(unique_items)), 2))
        arrayklant[:, 0] = unique_items
        for ding in range(len(unique_items)):
            countproduct = itemsalleklanten.count(itemsalleklanten[ding])
            # itemperklant.append(countproduct)
            arrayklant[ding, 1] = countproduct
        print(unique_items)
        print(arrayklant)
        sorted = arrayklant[np.argsort(arrayklant[:, 1])]
        print('sorted', sorted)
        product = np.array([])
        print('average length', top_nc[s])
        for reverse in range(int(top_nc[s])):
            print   ('test', sorted[-reverse - 1, :])
            product = np.append(product, sorted[-reverse, :])

    #print("uncommon products")
    #all_baskets = remove_products_which_are_uncommon(all_baskets)
    print("short baskets")
    medium_baskets, all_baskets = remove_short_baskets(all_baskets)
    #print(medium_baskets)

    print(len(medium_baskets))
    print(len(all_baskets))
    #print("nested change")
    #all_baskets = nested_change(all_baskets, embedding_wrapper.lookup_ind_f)
    #print("split_data")
    train_ub, val_ub_input, val_ub_target, test_ub_input, test_ub_target = split_data(all_baskets)
    #print('knndtw')
    #knndtw = KnnDtw(n_neighbors=[5])
    #preds_all, distances = knndtw.predict(train_ub, val_ub_input, embedding_wrapper.basket_dist_EMD, embedding_wrapper.basket_dist_REMD)
    #print(preds_all)
    #print(distances)
    #print("Wasserstein distance", sum(distances)/len(distances))
    #return preds_all, distances
    write_path = 'data/testprint'
    with open(write_path + '.txt', 'w') as results:
        results.write('All baskets test ' + str(all_baskets) + '\n')
    results.close()

def get_top_nc(all_baskets, iterations):
    top_nc_all = []
    for c in range(iterations):

        sizes = []
        for b in range(len(all_baskets[c])):
            sizes.append(len(all_baskets[c][b]))
        top_nc = sum(sizes) / len(sizes)
        top_nc_all.append(top_nc)
    return top_nc_all


def association_rules():
    embedding_wrapper = EmbeddingWrapper('product')
    bc = BasketConstructor('./data/', './data/')
    ub_basket = bc.get_baskets('prior', reconstruct=False)
    ok, ub_basket = train_test_split(ub_basket, test_size=0.20, random_state=0)
    # embedding_wrapper = EmbeddingWrapper('tafeng_products')
    # print(ub_basket)

    all_baskets = ub_basket.basket.values
    # print(all_baskets)
    # changes every item to string
    print("nested change")
    all_baskets = nested_change(list(all_baskets), str)
    print("embedding_wrapper.remove_products_wo_embeddings(all_baskets)")
    all_baskets = embedding_wrapper.remove_products_wo_embeddings(all_baskets)


if __name__ == "__main__":
    runtopncustomers()
