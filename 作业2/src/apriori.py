import pandas as pd
from optparse import OptionParser
import os


def run_apriori(user_behavior_record, min_support, min_confidence):
    purchase_record = filter_purchase_record(user_behavior_record)
    transaction = construct_transaction(purchase_record)
    one_item_list = get_1_item_set(user_behavior_record)
    support_list = []
    candidate_item_set_list = []
    frequent_item_set_list = []
    candidate_item_set_list.append(one_item_list)
    support = {tuple([int(item)]): 0 for item in candidate_item_set_list[0]}
    for t in transaction['items']:
        items = set(t.split(','))
        for item in items:
            support[tuple([int(item)])] += 1
    length = 1
    while True:
        length += 1
        support = {k: v for k, v in support.items() if v >= min_support}
        support_list.append([(k, v) for k, v in support.items()])
        frequent_item_set_list.append([x for x in support.keys()])
        candidate_item_set_list.append(generate_candidate_item_set(frequent_item_set_list[-1], length))
        if len(candidate_item_set_list[-1]) == 0:
            break
        transaction = transaction[transaction.apply(lambda x: len(x['items'].split(',')) >= length, axis=1)]
        support = {tuple(sorted(item)): 0 for item in candidate_item_set_list[-1]}
        for t in transaction['items']:
            items = set(int(x) for x in t.split(','))
            for k, v in support.items():
                if set(k).issubset(items):
                    support[k] += 1
    return support_list, get_confidence(support_list, min_confidence)


def get_confidence(support_list, confidence):
    """
    Return the confidence rules. confidence(pre->post)=support(pre+post)/support(pre)
    :param support_list: [((item_list), support)]
    :param confidence: minimum confidence
    :return: rules: [((pre, post), confidence)]
    """
    result = []
    for item_list in support_list[1:][::-1]:
        for support_item in item_list:
            rules = generate_rules(support_item)
            for pre, post in rules:
                pre_length = len(pre) - 1
                pre_support = [x for x in filter(lambda x: x[0] == pre, support_list[pre_length])][0][1]
                support_item_list_support = [x for x in filter(
                    lambda x: x[0] == support_item[0], support_list[len(support_item[0]) - 1])][0][1]
                this_confidence = float(support_item_list_support) / pre_support
                if this_confidence >= confidence:
                    result.append(tuple([tuple([pre, post]), this_confidence]))
    return result


def generate_rules(items):
    """
    Guarantee that len(pre)+len(post)=len(items)
    :param items: list of frequent items
    :return: [(pre, post)]
    """
    result = []
    support_item = items[0]
    import itertools
    for length in range(1, len(support_item)):
        pre_list = itertools.combinations(support_item, length)
        for pre_item in pre_list:
            post_item = tuple(set(support_item) - set(pre_item))
            result.append((tuple(sorted(pre_item)), tuple(sorted(post_item))))
    return result


def generate_candidate_item_set(item_list, length):
    """
    Generate candidate item list whose length is specified
    :param item_list: a list of frequent item set
    :param length: each frequent-item-set's length
    :return: candidate item set
    """
    if not isinstance(item_list, list):
        return None
    result = []
    for index1 in range(len(item_list) - 1):
        for index2 in range(index1 + 1, len(item_list)):
            union = set(item_list[index1]).union(set(item_list[index2]))
            if len(union) == length:
                result.append(tuple(union))
    return result


def get_1_item_set(user_behavior_record):
    return list(set(user_behavior_record['item_id'].astype('str')))


def filter_purchase_record(user_behavior_record):
    purchase = user_behavior_record[user_behavior_record['action_type'].isin([2])]
    selected = ['user_id', 'item_id', 'time_stamp']
    purchase = purchase.filter(items=selected)
    print('Finish filtering purchase records, size: {} lines of record.'.format(len(purchase)))
    return purchase.sort_values(by=['time_stamp'])


def construct_transaction(purchase, transaction_file='data_set/transaction.csv'):
    if transaction_file and os.path.exists(transaction_file):
        return pd.read_csv(transaction_file)
    groups = purchase.groupby(['time_stamp', 'user_id'])  # transaction = same user buys items in same day
    columns = ['user_id', 'time_stamp', 'items']
    transaction = pd.DataFrame(columns=columns)
    for t in groups:
        items = t[1]
        transaction = transaction.append(
            {
                'user_id': items['user_id'].astype('str').values[0],
                'items': ','.join(items['item_id'].astype('str').values),
                'time_stamp': items['time_stamp'].astype('str').values[0]
            }, ignore_index=True)
    transaction.to_csv('data_set/transaction.csv', index=False)
    print('Finish constructing transaction record, size: {} lines of record'.format(len(transaction)))
    return transaction


def generate_result_string(support_list, rules):
    result = ['------------------ FREQUENT ITEM SET ------------------']
    for i in range(0, len(support_list)):
        length = i + 1
        result.append('Item set length: {}'.format(length))
        for item, support in support_list[i]:
            result.append('\titem: {}, sup: {}'.format(item, support))
    result.append("\n------------------------ RULES ------------------------")
    for rule_item in rules:
        rule, confidence = rule_item
        pre, post = rule
        result.append("rule: {} ==> {}, confidence: {}".format(pre, post, confidence))
    return '\n'.join(result)


def write_to_file(result, destination='result/apriori_result.txt'):
    if not os.path.exists('result'):
        os.makedirs('result')
    with open(destination, 'w') as f:
        f.write(result)


if __name__ == '__main__':
    opt_parser = OptionParser()
    opt_parser.add_option('-f', '--input_file',
                          dest='input',
                          help='user behavior record csv',
                          default='data_set/user_log_format1.csv')
    opt_parser.add_option('-s', '--min_support',
                          dest='min_support',
                          help='minimum support value',
                          default=3,
                          type='int')
    opt_parser.add_option('-c', '--min_confidence',
                          dest='min_confidence',
                          help='minimum confidence value',
                          default=0.6,
                          type='float')
    opt_parser.add_option('-t', '--truncate_log_size',
                          dest='truncate_log_size',
                          help='truncate log size',
                          default=1024 * 1024 * 16,
                          type='int')
    (options, args) = opt_parser.parse_args()

    user_behavior = pd.read_csv(options.input)
    print('Finish reading user behavior file, size: {} lines of record.'.format(len(user_behavior)))
    user_behavior.sample(frac=1)
    user_behavior = user_behavior[:options.truncate_log_size]
    print('Finish shuffling and truncating user behavior record, size: {} lines of record.'.format(len(user_behavior)))
    support_item_list, confidence_rules = run_apriori(user_behavior, options.min_support, options.min_confidence)
    result_string = generate_result_string(support_item_list, confidence_rules)
    print(result_string)
    write_to_file(result_string)
