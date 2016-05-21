import scrape_bt2

con = open("test.html").read()
line = scrape_bt2.extract(con)
assert len(line.split(";")) == 8
