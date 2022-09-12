from rabin_origin.utils import RabinFingerprint, RsyncChecksum
import timeit
# import cProfile, pstats, StringIO
from collections import deque
from array import array


# pr = cProfile.Profile()

rf = RabinFingerprint(3)
# rf.feed_many('abr', is_str=True)
rf.feed_many([97, 98, 114])
def rabin_test():
	# rf.feed_many('a', is_str=True)
	# rf.feed_many([97])
	rf.feed(97)
	# ord('a')

rc = RsyncChecksum(3)
# rc.feed_many('abr', is_str=True)
# rc.feed_many([97, 98, 114])
rc.feed(97)
rc.feed(98)
rc.feed(114)
def rsync_test():
	# rc.feed_many('a', is_str=True)
	# rc.feed_many([97])
	rc.feed(97)

if __name__ == '__main__':
	# pr.enable()
	# print(timeit.timeit(stmt='rabin_test()', setup='from __main__ import rabin_test', number=1000000))
	# pr.disable()
	# s = StringIO.StringIO()
	# sortby = 'cumulative'
	# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	# ps.print_stats()
	# print s.getvalue()
	# cProfile.run('rabin_test()')
	# print(timeit.timeit('"h = (((999509 - (97 * 10201)) * 101) + 97)"', number=10000000))

	# pr.enable()
	# print(timeit.timeit(stmt='rsync_test()', setup='from __main__ import rsync_test', number=1000000))
	ct = RabinFingerprint(4)
	# ct.feed_many("jisd", is_str=True)
	ct.feed_many("asdf", is_str=True)
	print(ct.hash)
	# pr.disable()
	# s = StringIO.StringIO()
	# sortby = 'cumulative'
	# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	# ps.print_stats()
	# print s.getvalue()

	# a = deque(maxlen=3)
	# print(timeit.timeit(stmt='a.append(97)', setup='from __main__ import a', number=1000000))