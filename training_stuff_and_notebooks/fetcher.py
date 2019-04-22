import io
import sys
import json
import vk_fetcher

#VK_GROUPS = ['dobriememes']

v = vk_fetcher.VkFetcher()

def fetch_comments(domain):
    c = 0
    for i, posts_batch in enumerate(v.get_all_posts_id_pairs(domain)):
        comments_in_batch = []
        for post in posts_batch:
            comments = []
            for comments_batch in v.get_all_comments(post['owner_id'], post['id']):
                comments += comments_batch 
            c += len(comments)
            comments_in_batch += [{'post': post['id'], 'comments': comments, 'owner_id': post['owner_id']}]

        with io.open('results2/comments_{}_{}.json'.format(domain, i), 'w', encoding='utf8') as f:
            json.dump(comments_in_batch, f, indent=4, ensure_ascii=False)
            print(len(comments_in_batch), c)

def main(vk_groups):
    # test all groups
    for group in vk_groups:
        v.get_posts_id_pairs(group, 1, 0)
    print('groups checked: ok')

    for group in vk_groups:
        print('*' * 40)
        print('start fetching comments from', group)
        try:
            fetch_comments(group)
        except Exception as ex:
            print('failed:', ex.message)

if __name__ == '__main__':
    if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
        print('usage: fetch.py group_domain_1 group_domain_2 ...\ngroup_domain_i can be dobriememes')
    else:
        main(sys.argv[1:])
