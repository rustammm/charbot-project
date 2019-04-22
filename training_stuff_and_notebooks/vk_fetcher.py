import vk

# sensetive info
ACCESS_TOKEN = ''
STEP = 100

class VkFetcher:
    def __init__(self, access_token=ACCESS_TOKEN):
        self._session = session = vk.Session(access_token=ACCESS_TOKEN)
        self._api = vk.API(session, v='5.35')

    def get_post_raw(self, domain, count, offset):
        return self._api.wall.get(domain=domain,count=count, offset=offset)   

    def get_comments_raw(self, owner_id, post_id, count, offset, sort):
        return self._api.wall.getComments(owner_id=owner_id, post_id=post_id, count=count, offset=offset, sort=sort)

    def get_posts_id_pairs(self, domain, count, offset):
        content = self.get_post_raw(domain, count, offset)
        count = content['count']
        pairs = []
        for item in content['items']:
            pairs += [{'id': item['id'], 'owner_id': item['owner_id']}]
        return count, pairs

    def get_all_posts_id_pairs(self, domain):
        left_count, pairs = self.get_posts_id_pairs(domain=domain, count=STEP, offset=0)
        left_count -= len(pairs)
        yield pairs

        offset = 0
        while left_count > 0:
            offset=offset + STEP
            _, pairs = self.get_posts_id_pairs(domain, count=STEP, offset=offset)
            left_count -= len(pairs)
            yield pairs

    def get_comments(self, owner_id, post_id, count, offset, sort):
        content = self.get_comments_raw(owner_id, post_id, count, offset, sort)
        count = content['count']
        for item in content['items']:
            if 'attachments' in item:
                item.pop('attachments')
                if 'text' in item:
                    item['text'] = item['text'].encode('utf-8').decode('utf-8', 'ignore')
        return count, content['items']

    def get_all_comments(self, owner_id, post_id, sort='asc'):
        left_count, comments = self.get_comments(owner_id=owner_id, post_id=post_id, count=STEP, offset=0, sort=sort)
        left_count -= len(comments)
        yield comments

        offset = 0
        while left_count > 0:
            offset=offset + STEP
            _, comments = self.get_comments(owner_id=owner_id, post_id=post_id, count=STEP, offset=offset, sort=sort)
            left_count -= len(comments)
            yield comments
