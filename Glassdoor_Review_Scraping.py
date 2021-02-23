from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

import time
import datetime 
import pandas as pd

import sqlite3

# This function signs into the Glassdoor website. 
# After sign-in we are on Glassdoor homepage (https://www.glassdoor.com/member/home/index.htm)
def initialize():
    
    # Initialize webdriver using google chrome
    global driver #will be used in a variety of functions so we declare it as a global variable. 
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(executable_path="/Users/Shayan1/Desktop/DataProject/chromedriver", options=options)
    driver.set_window_size(1600, 1000)
    
    # Use webdriver to go to Glassdoor sign-in page
    url = 'https://www.glassdoor.com/profile/login_input.htm?userOriginHook=HEADER_SIGNIN_LINK'
    driver.get(url)
    
    # Click on text box in sign-in page for Email Address and Password
    try:
        #driver.find_element_by_class_name('css-q444d9').click()
        driver.find_element_by_class_name('css-1ohf0ui').click()
        #<div class=" "><label for="userEmail" class="css-2ku426"><span>Email Address</span></label><div class="input-wrapper css-q444d9"><input id="userEmail" name="username" title="Email Address" type="email" data-test="" aria-label="" class="css-qlknqs" value=""></div></div>
    except ElementClickInterceptedException:
        time.sleep(3.0)
        driver.find_element_by_class_name('css-q444d9').click()
        
    # Input personal information. Pause program for 1 second to avoid throtting Glassdoor server
    time.sleep(1.0)
    user_email = driver.find_element_by_id('userEmail')
    user_email.send_keys('sg538@cornell.edu') 
    
    time.sleep(1.0)
    user_password = driver.find_element_by_id('userPassword')
    user_password.send_keys('JoshAllen716') 
    
    # Click sign-in button
    time.sleep(1.0)
    #driver.find_element_by_class_name('gd-ui-button.minWidthBtn.css-1sdotxz').click()
    driver.find_element_by_class_name('gd-ui-button.minWidthBtn.css-8i7bc2').click()
    #<button class="gd-ui-button minWidthBtn css-8i7bc2" type="submit" name="submit">Sign In</button>
    
def get_reviews(company_name):
    
    local_df = pd.DataFrame(columns = ['company_name', 
                             'review_date', 
                             'review_header', 
                             'overall_rating', 
                             'work_life_balance', 
                             'culture_and_values', 
                             'diversity_and_inclusion', 
                             'career_opportunities', 
                             'compensation_and_benefits', 
                             'senior_management', 
                             'job_title', 
                             'job_location',
                             'recommends', 
                             'positive_or_negative_outlook',
                             'approves_of_ceo', 
                             'self_description', 
                             'pros', 
                             'cons', 
                             'advice_to_management']) 
    
    # Go to Glassdoor homepage for Companies
    time.sleep(1.0)
    url = 'https://www.glassdoor.com/member/home/companies.htm'
    driver.get(url)
    time.sleep(1.0)
    
    # Input company name into text box at top of webpage
    time.sleep(1.0)
    input_company_name = driver.find_element_by_id('sc.keyword')
    input_company_name.send_keys(company_name)
    
    # Input location for companies (United States)
    time.sleep(2.0)
    input_location = driver.find_element_by_id('sc.location')
    time.sleep(1.0)
    input_location.click()
    for i in range(15):
        time.sleep(.2)    
        input_location.send_keys(Keys.BACK_SPACE)
    time.sleep(1.0)
    input_location.send_keys('United States')
    time.sleep(1.0)
    
    # Click search button to search for Company
    time.sleep(1.0)
    driver.find_element_by_class_name('gd-ui-button.ml-std.col-auto.css-iixdfr').click()
    time.sleep(1.0)
    
    # Will click on the reviews button for the first company. 
    time.sleep(1.0)
    try:
       driver.find_element_by_class_name('ei-contribution-wrap.col-4.pl-lg-0.pr-0').click()
    except NoSuchElementException:
        pass

    try:
        #<a class="eiCell cell reviews " href="/Reviews/M-and-T-Bank-Reviews-E858.htm" data-label="Reviews"><span class="num h2"> 1.2k</span><span class="subtle"> Reviews</span></a>
       driver.find_element_by_class_name('eiCell.cell.reviews').click()
    except NoSuchElementException:
        pass                                                           
    time.sleep(20.0)
    
    '''
    # Now we will scrape all Glassdoor reviews for this company
    # Relevant data-fields per review are:
        # (1)  Review Date
        # (2)  Review header, Review Link
        # (3)  Overall Rating (1-5 scale)
        # (4)  Work/Life Balance (1-5 scale)
        # (5)  Culture & Values (1-5 scale)
        # (6)  Diversity & Inclusion (1-5 scale)
        # (7)  Career Opportunities (1-5 scale)
        # (8)  Compensation and Benefits (1-5 scale)
        # (9)  Senior Management (1-5 scale)
        # (10) Reviewer job title
        # (11) Reviewer Job Location
        # (12) Recommends? (y/n) 
        # (13) Positive/Negative outlook? [y/n]
        # (14) Approves of CEO? [y/n]
        # (15) Reviewer self description
        # (16) Pros (textual input)
        # (17) Cons (textual input)
        # (18) Advice to Management (textual input)
    '''
    
    keep_going = 'Next'
    i=1
    while (not driver.page_source):    
        time.sleep(1.0)
        print ('Time elapsed: ' + str(i))
        i = i+1
        
    while (keep_going == 'Next'):
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')    
        time.sleep(2.0)
        
        for tag in soup.find_all(class_='gdReview'):
            # (1)  Review Date
            review_date = tag.find(class_='d-flex align-items-center').text
            #print (' (1) Review Date: ' + review_date)
            
            # (2)  Review header, Review Link
            review_header = tag.find(class_='reviewLink').text
            #print (' (2) Review Header: ' + review_header)
            
            # (3)  Overall Rating (1-5 scale)
            overall_rating = tag.find(class_='v2__EIReviewsRatingsStylesV2__ratingNum v2__EIReviewsRatingsStylesV2__small').text
            #print (' (3) Overall Rating: ' + overall_rating)
            
            # Extract Information for Supplemental Ratings
                # (4) Work/Life Balance (1-5 scale)
                # (5) Culture & Values (1-5 scale)
                # (6) Diversity & Inclusion (1-5 scale)
                # (7) Career Opportunities (1-5 scale)
                # (8) Compensation and Benefits (1-5 scale)
                # (9) Senior Management (1-5 scale)
            supplemental_ratings = tag.find(class_='undecorated')
                
            work_life_balance = ''
            culture_and_values = ''
            diversity_and_inclusion = ''
            career_opportunities = ''
            compensation_and_benefits = ''
            senior_management = ''
        
            if (supplemental_ratings):
                attribute = supplemental_ratings.find_all(class_='minor')
                ratings = supplemental_ratings.find_all(class_='rating')
                if (len(attribute) == len(ratings)):
                    for i in range(len(attribute)):
                        if (ratings[i].find('span', {'title':'1.0'}) is not None):
                            rating = 1.0
                        if (ratings[i].find('span', {'title':'2.0'}) is not None):
                            rating = 2.0
                        if (ratings[i].find('span', {'title':'3.0'}) is not None):
                            rating = 3.0
                        if (ratings[i].find('span', {'title':'4.0'}) is not None):
                            rating = 4.0
                        if (ratings[i].find('span', {'title':'5.0'}) is not None):
                            rating = 5.0 
                    
                        if (attribute[i].text == 'Work/Life Balance'):
                            work_life_balance = rating
                        if (attribute[i].text == 'Culture & Values'):
                            culture_and_values = rating
                        if (attribute[i].text == 'Diversity & Inclusion'):
                            diversity_and_inclusion = rating
                        if (attribute[i].text == 'Career Opportunities'):
                            career_opportunities = rating
                        if (attribute[i].text == 'Compensation and Benefits'):
                            compensation_and_benefits = rating
                        if (attribute[i].text == 'Senior Management'):
                            senior_management = rating
        
            #print(' (4) Work/Life Balance: ' + str(work_life_balance))
            #print(' (5) Culture & Values: ' + str(culture_and_values))
            #print(' (6) Diversity & Inclusion: ' + str(diversity_and_inclusion))
            #print(' (7) Career Opportunities: ' + str(career_opportunities))
            #print(' (8) Compensation & Benefits: ' + str(compensation_and_benefits))
            #print(' (9) Senior Mangement: ' + str(senior_management))
            
            # (10) Reviewer job title
            if (tag.find('span', {'class':'authorJobTitle middle'}) is None):
                job_title = ''
            else:
                job_title = tag.find('span', {'class':'authorJobTitle middle'}).text
                #print('(10) Job Title: '+ job_title)
      
            # (11) Reviewer Job Location
            if (tag.find('span', {'class':'authorLocation'}) is None):
                job_location = ''
            else:
                job_location = tag.find('span', {'class':'authorLocation'}).text
            #print('(11) Job Location: ' + job_location)
          
            # (12) (13) (14) Get answers to [y/n] questions
            recommend = ''
            positive_or_negative_outlook = ''
            approves_of_ceo = ''
            if (tag.find(class_='row reviewBodyCell recommends') is None):
                recommend = ''
                positive_or_negative_outlook = ''
                approves_of_ceo = ''
                #print ('(12) Recommends: ' + recommend)
                #print ('(13) Positive/Negative outlook? [y/n]: ' + positive_or_negative_outlook)
                #print ('(14) Recommends: ' + approves_of_ceo)            
            else:
                yes_no_questions = tag.find(class_='row reviewBodyCell recommends').text
                # (12) Recommends? (y/n) 
                if ("Recommends" in yes_no_questions):
                    recommend = 'y'
                if ("Doesn't Recommend" in yes_no_questions):
                    recommend = 'n'
                    #print ('(12) Recommends: ' + recommend)
            
                # (13) Positive/Negative outlook? [y/n]
                if ("Positive Outlook" in yes_no_questions):
                    positive_or_negative_outlook = 'y'
                if ("Negative Outlook" in yes_no_questions):
                    positive_or_negative_outlook = 'n'
                    #print ('(13) Positive/Negative outlook? [y/n]: ' + positive_or_negative_outlook)
            
                # (14) Approves of CEO? [y/n]
                if ("Approves of CEO" in yes_no_questions):
                    approves_of_ceo = 'y'
                if ("Disapproves of CEO" in yes_no_questions):
                    approves_of_ceo = 'n'
                #print ('(14) Approves of CEO: ' + approves_of_ceo)
        
            # (15) Reviewer self description
            self_description = tag.find(class_='mainText mb-0').text
            #print ('(15) Self Description: ' + self_description)
    
            # (16) Pros 
            if (tag.find('span', {'data-test':'pros'}) is None):
                pros = ''
            else:
                pros = tag.find('span', {'data-test':'pros'}).text
                #print ('(16) Pros: ' + pros)
        
            # (17) Cons 
            if (tag.find('span', {'data-test':'cons'}) is None):
                cons = ''
            else:
                cons = tag.find('span', {'data-test':'cons'}).text
                #print ('(17) Cons: ' + cons)
        
            # (18) Advice to Management (textual input) 
            if (tag.find('span', {'data-test':'advice-management'}) is None):
                advice_to_management = ''
            else:
                advice_to_management = tag.find('span', {'data-test':'advice-management'}).text
                #print ('(18) Advice to Management: '+ advice_to_management)
                print ()
            
        
            #Add data into pandas dataframe. 
            new_review = {'company_name': company_name, 
                          'review_date': review_date, 
                          'review_header': review_header, 
                          'overall_rating': overall_rating, 
                          'work_life_balance': work_life_balance , 
                          'culture_and_values': culture_and_values, 
                          'diversity_and_inclusion': diversity_and_inclusion, 
                          'career_opportunities': career_opportunities, 
                          'compensation_and_benefits': compensation_and_benefits, 
                          'senior_management': senior_management, 
                          'job_title': job_title, 
                          'job_location': job_location, 
                          'recommends': recommend, 
                          'positive_or_negative_outlook': positive_or_negative_outlook,
                          'approves_of_ceo': approves_of_ceo, 
                          'self_description': self_description, 
                          'pros': pros, 
                          'cons': cons, 
                          'advice_to_management': advice_to_management}
        
            local_df = local_df.append(new_review, ignore_index=True)
            
        print (str(local_df.shape[0]) + " " + company_name)
        # Once we get to last page of reviews, we will no longer be able to click the 'Next'
        # button to get to the next page. Once that happens data-scraping reviews for this specific
        # company will be finished. 
        try:
            driver.find_element_by_class_name('pagination__ArrowStyle__nextArrow ').click()
            time.sleep(5.0)
        except ElementClickInterceptedException:
            keep_going = 'Exit'
        except NoSuchElementException:
            keep_going = 'Exit'
        
        # Once we get to 2,000 reviews for a specific company, terminate getting more reviews to
        # keep dataset of reasonable size. 
        if (local_df.shape[0]>=1000):
            keep_going = 'Exit'
        time.sleep(3.0)
          
    return local_df

def create_database():
    # Database is stored in same folder as this program. 
    con = sqlite3.connect('glassdoor_reviews.db')  
    c = con.cursor() 
    c.execute('''CREATE TABLE glassdoor_reviews_data (
                        [company_name] text,
                        [review_date] text,
                        [review_header] text,
                        [overall_rating] text,
                        [work_life_balance] text,
                        [culture_and_values] text,
                        [diversity_and_inclusion] text,
                        [career_opportunities] text,
                        [compensation_and_benefits] text,
                        [senior_management] text,
                        [job_title] text,
                        [job_location] text,
                        [recommends] text,
                        [positive_or_negative_outlook] text,
                        [approves_of_ceo] text,
                        [self_description] text,
                        [pros] text,
                        [cons] text,
                        [advice_to_management] text)''')

def add_reviews_to_database(local_df):
    con = sqlite3.connect('glassdoor_reviews.db')  
    local_df.to_sql(con = con, 
                    name = 'glassdoor_reviews_data', 
                    if_exists = 'append', 
                    index = False)
    #print (local_df)
    #print ('Done for Database')
    
#create_database()  
initialize()

# List of companies in the Nasdaq 100
company_list_1 = ['Activision Blizzard', 
                  'Adobe Inc', 
                  'Advanced Micro Devices', 
                  'Alexion Pharmaceuticals',
                  'Align Technologies',  
                  'Amazon',                 
                  'Amgen Inc',
                  'Analog Devices',
                  'Ansys',
                  'Apple',]
company_list_2 = ['Applied Materials',
                  'ASML',
                  'Autodesk',
                  'ADP',
                  'Baidu',
                  'Biogen',
                  'BioMarin Pharmaceutical',
                  'Booking.com', #Booking Holdings is a parent company and has only 6 reviews. Booking.com has appx. 4000 reviews. 
                  'Broadcom',
                  'Cadence Design Systems']
company_list_3 = ['CDW',
                  'Cerner Corporation',
                  'Spectrum', #Holding company is Charter Communications,
                  'Check Point Software Technologies',
                  'Cintas Corporation',
                  'Cisco Systems',
                  'Citrix Systems',
                  'Cognizant Technology Solutions',
                  'Comcast Corporation',
                  'Copart']
company_list_4 = ['Costco Wholesale',
                  'CSX Corporation',
                  'DexCom Inc',
                  'DocuSign',
                  'Dollar Tree Inc', #
                  'eBay Inc',
                  'Electronic Arts', #
                  'Exelon Corp',
                  'Expedia Group', #
                  'Facebook Inc']
company_list_5 = ['Fastenal Company', #
                  'Fiserv Inc', #
                  'Fox Broadcasting', # #Fox Corp is a conglomerate. Fox Broadcasting has the most number of reviews.  
                  'Gilead Sciences',
                  'Google', #  #Parent company is Alphabet.  
                  'IDEXX Laboratories',
                  'Illumina', #
                  'Incyte',
                  'Intel Corporation', #
                  'Intuit Inc'] #
company_list_6 = ['Intuitive Surgical',
                  'JD.com', #
                  'KLA Corporation',
                  'Kraft Heinz',
                  'Lam Research',
                  'Liberty Global',
                  'Lululemon',
                  'Marriott International',
                  'Maxim Integrated',
                  'MercadoLibre'] 
company_list_7 = ['Microchip Technology',
                  'Micron Technology',
                  'Microsoft',
                  'Moderna',
                  'Mondelez International',
                  'Monster Beverage',
                  'NetEase.com',
                  'Netflix',
                  'NVIDIA',
                  'NXP Semiconductors']
company_list_8 = ['O Reilly Auto Parts',
                  'PACCAR',
                  'Paychex',
                  'PayPal',
                  'PepsiCo',
                  'Pinduoduo', #
                  'Qualcomm', #
                  'Regeneron Pharmaceuticals', #
                  'Ross Stores',
                  'Seagen'] #Pinduoduo is a Chinese ADR
company_list_9 = ['SiriusXM', #
                  'Skyworks Solutions',
                  'Splunk Inc',
                  'Starbucks',
                  'Synopsys',
                  'T-Mobile',
                  'Take-Two',
                  'Tesla Inc',
                  'Texas Instrumenents', #
                  'Trip.com']
company_list_10 = ['Ulta Beauty',
                   'VeriSign',
                   'Verisk Analytics',
                   'Vertex Pharmaceuticals',
                   'Walgreens Boots Alliance',
                   'Western Digital',
                   'Workday Inc',
                   'Xcel Energy Inc',
                   'Xilinx Inc',
                   'Zoom Video Comm'] #

company_list_missing = [#'Dollar Tree Inc', 
                        #'Electronic Arts', #
                        #'Expedia Group', 
                        #'Fastenal Company', 
                        #'Fiserv Inc', 
                        #'Fox Broadcasting', 
                        #'Google', 
                        #'Illumina', 
                        #'Intel Corporation', 
                        #'Intuit Inc', 
                        #'JD.com', 
                        #'Pinduoduo', 
                        #'Qualcomm', 
                        #'Regeneron Pharmaceuticals', #
                        #'SiriusXM', 
                        'Texas Instruments'] #
                        #'Zoom Video Communications'] # #17
i=1
for company in company_list_missing:
    begin_time = datetime.datetime.now()
    df = get_reviews(company)
    add_reviews_to_database(df)
    print (datetime.datetime.now()-begin_time)
    print ('Company Number: ' + str(i))
    print ()
    i=i+1
    time.sleep(5.0)
    



