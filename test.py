from datetime import timedelta, datetime
from forex_python.converter import get_rate

start_date = datetime(2019,1,1)
end_date = datetime.now()

conversion_usd_gbp = []

for n in range(int ((end_date - start_date).days)):
	conversion_usd_gbp.append(get_rate('USD', 'GBP', (start_date + timedelta(n))))




print(conversion_usd_gbp)