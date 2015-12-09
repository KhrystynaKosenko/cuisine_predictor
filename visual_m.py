#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pylab as plt
import pandas
import time

matplotlib.style.use('ggplot')
start = time.time()
traindf = pandas.read_json('train.json')

# traindf = traindf.sort_values('cuisine', ascending=True)
# 
# ingredients = [len(x) for x in traindf['ingredients']]

unic_ingred = []
[unic_ingred.append(z) for x in traindf['ingredients'] for z in x if z not in unic_ingred]
cuisine = []
[cuisine.append(x) for x in traindf['cuisine'] if x not in cuisine]

stat_cuisine = pandas.DataFrame({
                            'cuisine': cuisine,
                            'most_used_ingredient': None,
                            'value': None,
                            'total_quantity_of_ingredients': None
    })

def cuisine_ingred_count(cuisine):
    traf = traindf[traindf['cuisine'].isin([cuisine])]
    ingredients = []
    [ingredients.append(z) for x in traf['ingredients']
                            for z in x]
    ingredient_counted = []
    ingredient_index = []
    [(ingredient_index.append(unic_ingred.index(x)),
        ingredient_counted.append(ingredients.count(x)))
            for x in unic_ingred
                if x not in ingredient_counted and ingredients.count(x) != 0]
    return ingredient_index, ingredient_counted

def cuisine_lines(cuisines):
    lines = []
    colors = [
            'bo-', 'co-', 'go-', 'mo-', 'ko-', 'ro-', 'yo-', 'wo-',
            'b^-', 'c^-', 'g^-', 'm^-', 'k^-', 'r^-', 'y^-', 'w^-',
            'bs-', 'cs-', 'gs-', 'ms-', 'ks-', 'rs-', 'ys-', 'ws-'
        ]
    for cuisine in cuisines:
        x, y = cuisine_ingred_count(cuisine)
        lines.append(x)
        lines.append(y)
        lines.append(colors[cuisines.index(cuisine)])
    return lines
plt_args = cuisine_lines(cuisine)
plt_args = str(plt_args)[1:-1]
eval('plt.plot(%s)' % plt_args)
# plt.plot(
#     plt_args[0], plt_args[1], plt_args[2],
#     plt_args[3], plt_args[4], plt_args[5],
#     plt_args[6], plt_args[7], plt_args[8],
#     plt_args[9], plt_args[10], plt_args[11],
#     plt_args[12], plt_args[13], plt_args[14]
#     )
plt.show()

# cuisine_plot = []
# [cuisine_plot.append(cuisine.index(x[0]) + 1) for x in zip(traindf['cuisine'], ingredients) for y in range(x[1])]
# ingredients_plot = []
# [ingredients_plot.append(unic_ingred.index(x) + 1) for z in traindf['ingredients'] for x in z]
# 

# ingred = [x for z in traindf['ingredients'] for x in z]
# count_ingred = [ingred.count(x) for x in unic_ingred]
# popin = pandas.DataFrame({'ingredient': unic_ingred,
#                           'quantity': count_ingred})
# # popular_ingred = sorted(popular_ingred, reverse=True)
# popin = popin.sort_values('quantity', ascending=False)
# print popin, len(count_ingred), len(unic_ingred)
# print time.time() - start


# print len(cuisine_plot), len(ingredients_plot)

# # ingredients = []
# # ingredients = []
# # [(cuisine.append(x[1]), ingredients.append(y)) for x in traindf[['ingredients', 'cuisine']] for y in x[0]]
# cuisine_axis = [(cuisine.index(x)+1) for x in traindf['cuisine']]

# plt.plot(ingredients_plot, cuisine_plot, 'ro')
# plt.axis([0, 7000, 0, 5])
# plt.show()

# [ingredients.append(z) for (x, z) in (traindf['ingredients'], traindf['cuisine']) for y in x]

# print cuisine[:10]