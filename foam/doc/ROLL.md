Decide your rolling policy

Depending on what kind of rolling pattern you have you need to
determine your rolling policy. The rolling policy is made up of two
components.

o How many days before expiry do we roll?
o Which subset of available contracts do we consider in our rolling cycle?

So for example take a case 1 contract (liquidity 'for a year') with
monthly expiries, of which only the quarterly IMM dates are liquid,
and where we want to avoid holding the front month perhaps so we can
always measure rolldown. Then we'd want to roll something like 45 days
before expiry (less than 31 means we'll end up in the undesirable
front), and only include the HMUZ monthly expiries in our roll cycle.

For an 'infinite' liquidity like Eurodollar I like to stay about 40
months out. At this point on the curve there are only quarterly
expiries anyway, and we'd want to roll about 1216 days before expiry.

If we're in an 'only the front' contract like Korean bonds then we'd
almost certainly include all contracts in the cycle. I'd probably want
to roll close to expiry, unless like US 10 year bonds liquidity comes
in earlier than that, or we're trying to avoid a first notice date.

In a seasonal case like Crude oil, since I'm sticking to the winter
flavour, I'd only include the 'Z' contract in my roll cycle. I want to
measure rolldown, and with monthly expiries I'll be doing so off the
November 'X' contract, so I need to roll at least 45 days before
expiry. Notice the items in red here.

Let's take a simple example. Suppose I had Gold prices from 1st
January 1980. To ensure I can measure rolldown I want to be in the
second contract, but no further. My roll cycle is GJMQVZ which are the
liquid months, all of which are around 2 months, 60 days, apart. To
make things simple let's ignore the possibility of having a front
month not in the roll cycle to measure rolldown off, and switch 70
days before expiry.

In January I'd want to be in April Gold, so I can measure rolldown off
February. 70 days before April Gold expires I would then switch to
June Gold. 70 days before June Gold expires I would move to August
Gold, and so on.

F - Jan, G - Feb, H - Mar, J - Apr, K - May, M - Jun
N - Jul, Q - Aug, U - Sep, V - Oct, W - Nov, Z - Dec
