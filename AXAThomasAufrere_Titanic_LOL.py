""
#Import section
import urllib
import re
import csv

#open csv
fname = "\\\NR5VFL01.hk.intraxa\\thomas.aufrere$\\Documents\\1. AARO\\Data Science\\test.csv"
file = open(fname, "rb")
reader = csv.reader(file)
result = []
ID = []

#loop 
for row in reader:
    print row[0],row[2]
     
    name = row[2]
    print name
    
    #open url
    sauce = urllib.urlopen('http://www.bing.com/search?q='+name).read()
    #find answer
    regex = "https://www.encyclopedia-titanica.org/titanic-(.+?)/"
    pattern = re.compile(regex)

    temp = re.findall(pattern,sauce)    
    if len(temp) > 0:
        temp = temp[0]   
    #add result
    if temp == 'victim': temp = 0
    else: temp = 1        
    result.append(temp)
    ID.append(row[0])
   
  
print result
print ID
file.close()

result = result[1:]
ID = ID[1:]

test_file = open("\\\NR5VFL01.hk.intraxa\\thomas.aufrere$\\Documents\\1. AARO\\Data Science\\AXAThomasAufrere_Titanic_lol.csv", "wb")
open_file_object = csv.writer(test_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(ID, result))
test_file.close()


