from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler
import os

save_dir = 'car_picture'

if not os.path.isdir('./downloads/%s' % save_dir):
    os.mkdir('./downloads/%s' % save_dir)

for keyword in ['car', '차']:
    filters = dict(
        size='>320x320', type='photo')

    bing_crawler = BingImageCrawler(downloader_threads=4,
                                    storage={'root_dir': './downloads/%s' % save_dir})
    bing_crawler.crawl(keyword=keyword, filters=filters, offset=0, max_num=3000,
                       file_idx_offset='auto')
