import imaplib, email 
import time
import datetime 
from binance.client import Client
import pandas as pd
import sqlite3
import math

# Connect to gmail account and go to Inbox
FROM_EMAIL  = "kdrogo278@gmail.com"
FROM_PWD    = "ezevjwokbrdsldmi"
SMTP_SERVER = "imap.gmail.com" # "imap.googlemail.com"
SMTP_PORT   = 993

mail = imaplib.IMAP4_SSL(SMTP_SERVER)
mail.login(FROM_EMAIL,FROM_PWD)

# Login info to connect to Binance
api_key = ''
api_secret = ''
client = Client(api_key, api_secret)
  
# This function reads gmail account for all unread messages. Trading view sends 
#   alerts to this account anytime a trend has flipped based on EMASAR data
def readmail():   
    
    mail = imaplib.IMAP4_SSL(SMTP_SERVER, SMTP_PORT)
    mail.login(FROM_EMAIL,FROM_PWD)
    
    # Go to email inbox and read all unread messages
    mail.select('Inbox')
    try:
        typ, data = mail.search(None, '(UNSEEN)')
    except imaplib.IMAP4.abort:
        print("IMAP abort")
    
    trading_alerts = []
    
    # All relevant data is in email header. Extract data from email header.  
    for num in data[0].split():
        try: 
            typ, data = mail.fetch(num, '(RFC822)') # local variable 'data'
        except imaplib.IMAP4.abort:
            print("IMAP abort")
        
        raw_email = data[0][1]
        raw_email_string = raw_email.decode('utf-8')
        email_message = email.message_from_string(raw_email_string)
        email_subject = email_message['subject']
        email_subject = email_subject.replace("\r\n","")
        split = email_subject.split(", ")
        trading_alerts.append(split)
    
    return trading_alerts

# Determine size of opening position based on ticker daily volume
def position_size(symbol, position_size):
    # Max position size is defined as thrice the average volume transacted per second
    max_position_size = (2*round(float(client.futures_ticker(symbol = symbol)['quoteVolume'])/(24*60*60), 0))
    if (position_size>max_position_size):
        return max_position_size
    else:
        return position_size

# Determine percent of capital to allocate for trade
def capital_allocation(symbol, long_or_short):
    # Login info to connect to Binance Bot 1
    key = 'HJw3P8MUWXMHL7q21NaA7Hj0R5A3aA6cAbpzi28cOZcwFP6K21DhHsDToXFDFqb6'
    secret = '6tJDLPX7E34J0Tnfjkc7LRYXGhQaefg8Ib8NXyz3ji2MHNNrFjghr4oyax2o6euz'
    client_bot1 = Client(key, secret)
    
    position_information = client_bot1.futures_position_information()
    if (long_or_short == 'Long'):
        long_unrealizedPnl   = 0
        num_long_positions   = 0
        
        for pos in position_information:
            if (float(pos['positionAmt'])>0):
                long_unrealizedPnl = long_unrealizedPnl +(float(pos['unRealizedProfit'])/float(pos['notional']))*10000
                num_long_positions = num_long_positions+1
        
        if (num_long_positions == 0):
            long_unrealizedPnl = 0
        else: 
            long_unrealizedPnl = long_unrealizedPnl/num_long_positions
        
        if (long_unrealizedPnl<-25):
            print ('Long Unrealized:', long_unrealizedPnl, '0.00')
            return 0.0 
        elif (-25<long_unrealizedPnl and long_unrealizedPnl<=50):
            print ('Long Unrealized:', long_unrealizedPnl, '0.015')
            return .015
        elif (50<long_unrealizedPnl and long_unrealizedPnl<=100):
            print ('Long Unrealized:', long_unrealizedPnl, '0.012')
            return .012
        elif (100<long_unrealizedPnl):
            print ('Long Unrealized:', long_unrealizedPnl, '0.01')
            return .01
        
    if (long_or_short == 'Short'):
        short_unrealizedPnl  = 0
        num_short_positions  = 0
        for pos in position_information:
            if (float(pos['positionAmt'])<0):
                short_unrealizedPnl = short_unrealizedPnl +(float(pos['unRealizedProfit'])/(-1*float(pos['notional'])))*10000
                num_short_positions = num_short_positions+1
        
        if (num_short_positions == 0):
            short_unrealizedPnl = 0
        else: 
            short_unrealizedPnl = short_unrealizedPnl/num_short_positions
        
        if (short_unrealizedPnl<-25):
            print ('Short Unrealized:', short_unrealizedPnl, '0.00')
            return 0.0
        elif (-25<short_unrealizedPnl and short_unrealizedPnl<=50):
            print ('Short Unrealized:', short_unrealizedPnl, '0.015')
            return .015
        elif (50<short_unrealizedPnl and short_unrealizedPnl<=100):
            print ('Short Unrealized:', short_unrealizedPnl, '0.012')
            return .012
        elif (100<short_unrealizedPnl):
            print ('Short Unrealized:', short_unrealizedPnl, '0.01')
            return .01
    
    return 0.0

def get_significance_level(symbol):
    if (symbol=='BTCUSDT' or symbol=='YFIUSDT'  or symbol=='YFIIUSDT'or symbol=='ETHUSDT' or 
        symbol=='BCHUSDT' or symbol=='LTCUSDT'  or symbol=='XMRUSDT' or symbol=='DASHUSDT' or 
        symbol=='ZECUSDT' or symbol=='COMPUSDT' or symbol=='MKRUSDT' or symbol=='DEFIUSDT'):
        significance_level = 3
    elif (symbol=='ETCUSDT' or symbol=='LINKUSDT' or symbol=='BNBUSDT' or 
          symbol=='ATOMUSDT' or symbol=='NEOUSDT'):
        significance_level = 2
    elif (symbol=='1INCHUSDT' or symbol=='ADAUSDT'   or symbol=='ALPHAUSDT' or symbol=='AVAXUSDT' or 
          symbol=='AXSUSDT'   or symbol=='BELUSDT'   or symbol=='BLZUSDT'   or symbol=='BZRXUSDT' or
          symbol=='CTKUSDT'   or symbol=='CVCUSDT'   or symbol=='DOGEUSDT'  or symbol=='ENJUSDT'  or
          symbol=='FLMUSDT'   or symbol=='FTMUSDT'   or symbol=='GRTUSDT'   or symbol=='HNTUSDT'  or
          symbol=='ICXUSDT'   or symbol=='IOSTUSDT'  or symbol=='KNCUSDT'   or symbol=='LRCUSDT'  or
          symbol=='MATICUSDT' or symbol=='NEARUSDT'  or symbol=='OCEANUSDT' or symbol=='RENUSDT'  or
          symbol=='RSRUSDT'   or symbol=='RUNEUSDT'  or symbol=='SKLUSDT'   or symbol=='SOLUSDT'  or
          symbol=='SRMUSDT'   or symbol=='STORJUSDT' or symbol=='SUSHIUSDT' or symbol=='TOMOUSDT' or
          symbol=='TRXUSDT'   or symbol=='UNIUSDT'   or symbol=='VETUSDT'   or symbol=='XLMUSDT'  or
          symbol=='ZILUSDT'):
        significance_level = 0
    else:
        significance_level = 1    
    
    return significance_level

# This fuction makes opening trade at the mark of the symbol being traded. 
#    Once trade is entered, a take_profit and stop_loss order are created. 
def make_opening_trade(symbol, long_or_short):
    
    # Adjust leverage level
    if (long_or_short == 'Long'):    
        leverage = 5    
    if (long_or_short == 'Short'):
        leverage = 5
        
    client.futures_change_leverage(symbol = symbol, leverage=leverage)
    
    # Dynamically determine quantity traded based on unrealized profit on Bot 1
    percent_capital = capital_allocation(symbol = symbol, long_or_short = long_or_short)
    price  = float(client.futures_symbol_ticker(symbol = symbol).get('price'))
    # Binance throws an error if you enter a trade with too much specificity
    #   i.e. a trade on GRT for 30.3 shares will throw an error, but 30 is okay.  
    # For BTCUSDT, YFIUSDT, make trades as little as .001
    # For GRTUSDT, trades can be only be round numbers
    significance_level = get_significance_level(symbol = symbol)
    
    # Quantity determined on a percentage basis of capital
    percent_of_capital = (percent_capital*
                         (float(client.futures_account_balance()[0].get('withdrawAvailable'))/
                              float(client.futures_account_balance()[0].get('balance'))))
    balance  = float(client.futures_account_balance()[0].get('balance'))
    #quantity = round((balance/price)*percent_of_capital*leverage, significance_level)
    quantity = round(position_size(symbol = symbol,
                                   position_size=(percent_of_capital*balance*leverage))/price, 
                     significance_level)
    
    # Enter trade at the last price traded for a specific ticker
    price  = float(client.futures_symbol_ticker(symbol = symbol).get('price'))
    
    print('Opening Trade: ', symbol, str(quantity), long_or_short)
    
    # Do not make trade if quantity traded is rounded down to zero. 
    if (quantity<=0.0):
        return
    
    # Open position. 
    if (long_or_short == 'Short'):
        side = 'SELL'
    if (long_or_short == 'Long'):
        side = 'BUY'
    # Continuously make GTC orders at the 
    #   last price to enter in the trade
    orderId = client.futures_create_order(
                    symbol = symbol,
                    side = side,
                    type = 'LIMIT', 
                    timeInForce = 'GTC',
                    quantity = quantity, 
                    price=price)
    
    # Give binance 5.0 seconds to fill the trade. 
    time.sleep(5.0)
    client.futures_cancel_all_open_orders(symbol = symbol)
    
    # If trade was executed, executed_quantity > 0. 
    #             Otherwise, executed_quantity = 0
    executed_quantity = float(client.futures_get_order(
                                        symbol=symbol, 
                                        orderID = orderId.get('orderId')).get('executedQty'))
    
    # If initial trade was not executed, execute while loop for 
    #   until trade is executed up to 100 iterations. 
    num_attempts = 0
    while(executed_quantity==0.0 and num_attempts<10):
        price = float(client.futures_symbol_ticker(symbol = symbol).get('price'))
        orderId = client.futures_create_order(
                        symbol = symbol,
                        side = side,
                        type = 'LIMIT', 
                        timeInForce = 'GTC',
                        quantity = quantity, 
                        price=price)
        
        # Give binance 3.0 seconds to fill the trade. 
        time.sleep(3.0)
        client.futures_cancel_all_open_orders(symbol = symbol)
        executed_quantity = float(client.futures_get_order(
                                        symbol=symbol, 
                                        orderID = orderId.get('orderId')).get('executedQty'))
        print ("Num Attempts: ", str(num_attempts), 'Price: ', price)
        if (executed_quantity == 0):    
            num_attempts = num_attempts+1
            
    # If trade was not executed after 100 attempts, no longer attempt trade. 
    #   If trade needs to be executed, this is where we can add a market order. 
    if(num_attempts>=10):
        return
    
    return client.futures_get_order(symbol=symbol, 
                                    orderID = orderId.get('orderId'))
        
def make_closing_trade(symbol, long_or_short, quantity):
    print (symbol, long_or_short, quantity)
    quantity = float(quantity)
    notional = float(client.futures_symbol_ticker(symbol = symbol).get('price'))*quantity
 
    # If quantity is 0 or notional value less than 1.0
    if (quantity == 0 or notional<=.5):
        return None
    if (float(client.futures_position_information(symbol=symbol)[0]['positionAmt'])==0.0):
        return
    
    # Close position.
    if (long_or_short == 'Long'):
        side = 'BUY'
    if (long_or_short == 'Short'):
        side = 'SELL'
   
    executed_quantity = 0
    num_attempts = 1
    while(executed_quantity<quantity and num_attempts<10):
        price = client.futures_symbol_ticker(symbol = symbol).get('price')
       
        orderId = client.futures_create_order(
                    symbol = symbol,
                    side = side,
                    type = 'LIMIT',
                    timeInForce = 'GTC',
                    quantity = round(quantity-executed_quantity, get_significance_level(symbol = symbol)),
                    reduceOnly = True, 
                    price=price)
       
        # Give order 3 seconds to execute and .25 seconds for cancel all orders to hit exchange.
        time.sleep(3)
        client.futures_cancel_all_open_orders(symbol = symbol)
        time.sleep(.25)
   
        executed_quantity = round((executed_quantity + 
                                   float(client.futures_get_order(
                                          symbol=symbol,
                                          orderID = orderId.get('orderId')).get('executedQty'))), 
                                  get_significance_level(symbol = symbol))
                                  
       
        print ('Attempt #',
               num_attempts,
               orderId['symbol'],
               orderId['side'],
               orderId['price'],
               round(quantity-executed_quantity, get_significance_level(symbol = symbol)),
               executed_quantity)
       
        if (executed_quantity == 0):    
            num_attempts = num_attempts+1
        if (executed_quantity > 0 and executed_quantity<quantity):
            print ('PARTIAL FILL')
   
    # If trade was not executed after certain number of attempts, enter market order
    if(executed_quantity<quantity):
        orderId = client.futures_create_order(
                    symbol = symbol,
                    side = side,
                    type = 'MARKET',
                    quantity = quantity-executed_quantity)
        time.sleep(1.0)
   
    closing_trade = client.futures_get_order(symbol=symbol,
                                             orderID = orderId.get('orderId'))
    closing_trade['executedQty'] = quantity
   
    return closing_trade

def close_hanging_positions(all_positions):
    num_hanging_positions = 0
    for pos in client.futures_position_information():
        # Check to see if position is locally saved. 
        if (float(pos['positionAmt'])!=0):
            position_locally_saved = False
            for locally_saved_pos in all_positions:
                if (pos['symbol']==locally_saved_pos['symbol']):
                    position_locally_saved = True
                     
            # If position is not locally saved, make a GTC order for 2 seconds to close position        
            if (not position_locally_saved):
                num_hanging_positions = num_hanging_positions+1
                if (float(pos['positionAmt'])>0):
                    orderId = client.futures_create_order(
                                symbol = pos['symbol'],
                                side = 'SELL',
                                type = 'LIMIT', 
                                timeInForce = 'GTC',
                                quantity = pos['positionAmt'],
                                reduceOnly = True,
                                price=float(client.futures_symbol_ticker(symbol = pos['symbol']).get('price')))
                    time.sleep(3.0)
                    client.futures_cancel_all_open_orders(symbol = pos['symbol'])
                    print ('Hanging Position Closing Trade: ', 
                           orderId['symbol'], 
                           pos['notional'], 
                           orderId['price'])
        
                if (float(pos['positionAmt'])<0):
                    orderId = client.futures_create_order(
                                symbol = pos['symbol'],
                                side = 'BUY',
                                type = 'LIMIT', 
                                timeInForce = 'GTC',
                                quantity = pos['positionAmt'][1:],
                                reduceOnly = True,
                                price=float(client.futures_symbol_ticker(symbol = pos['symbol']).get('price')))
                    time.sleep(3.0)
                    client.futures_cancel_all_open_orders(symbol = pos['symbol'])
                    print ('Hanging Position Closing Trade: ', 
                           orderId['symbol'], 
                           pos['notional'],  
                           orderId['price'])
    
    return num_hanging_positions

def add_stop_loss_level(position):
    
    ticker_info     = client.futures_ticker(symbol = position.get('symbol'))
    current_price   = float(ticker_info.get('lastPrice'))
    position['max_profit_price'] = current_price
    
    if (position.get('side')=='BUY'):
        inital_stop_loss = .02
        position['current_price']        = round(current_price, 4)
        position['stop_loss_threshold']  = round(position['current_price']*inital_stop_loss, 6)
        position['stop_loss_level']      = round(position['current_price']-
                                                 position['stop_loss_threshold'], 4)
        
    if (position.get('side')=='SELL'):
        inital_stop_loss = .02
        position['current_price']        = round(current_price, 4)
        position['stop_loss_threshold']  = round(position['current_price']*inital_stop_loss, 6)
        position['stop_loss_level']      = round(position['current_price']+
                                                 position['stop_loss_threshold'], 4)
        
    return position

def add_take_profit_level(position, take_profit):
    if(position['side']=='BUY'):
        position['take_profit_level'] = float(position['avgPrice'])*1.01
    if(position['side']=='SELL'):
        position['take_profit_level'] = float(position['avgPrice'])*.99
    
    return position
    
def update_stop_loss_level(position):
    # Update stop-loss level
    position['current_price'] = round(float(
        client.futures_ticker(symbol = position.get('symbol')).get('lastPrice')), 5)
    
    # If we are at a 1% gain for longs and 1% for shorts, increase stop-loss price to break-even level
    if (position.get('side')=='BUY' and 
        position['current_price']/float(position['avgPrice'])>1.01 and 
        position['stop_loss_level']<float(position['avgPrice'])):
        position['stop_loss_threshold'] = round(position['current_price']-float(position['avgPrice']), 7)
    if (position.get('side')=='SELL' and 
        position['current_price']/float(position['avgPrice'])<.99 and 
        position['stop_loss_level']>float(position['avgPrice'])):
        position['stop_loss_threshold'] = -1*round(position['current_price']-float(position['avgPrice']), 7)
    
    # If current price breaks through max profit price, increase stop loss threshold by 60% the 
    # difference in current price and max profif price. 
    if (position.get('side')=='BUY' and 
        position['current_price']/float(position['avgPrice'])>1.01 and 
        position['current_price']>position['max_profit_price'] and
        position['stop_loss_level']>float(position['avgPrice'])):
        position['stop_loss_threshold'] = (position['stop_loss_threshold']+
                                          (position['current_price']-position['max_profit_price'])*.35)
    if (position.get('side')=='SELL' and 
        position['current_price']/float(position['avgPrice'])<.99 and 
        position['current_price']<position['max_profit_price'] and 
        position['stop_loss_level']<float(position['avgPrice'])):
        position['stop_loss_threshold'] = (position['stop_loss_threshold']+
                                          (position['max_profit_price']-position['current_price'])*.35)
    
    # If position is at a loss, move stop-loss up aggressively. 
    # If position is at a gain, move stop-loss up gradually. 
    position['stop_loss_threshold'] = round(position['stop_loss_threshold']*.99, 7)
    
    # Update maximum profit price, stop-loss level
    if (position.get('side')=='BUY'):
        if (position['current_price']>position['max_profit_price']):
            position['max_profit_price'] = position['current_price']
        if (position['max_profit_price']-position['stop_loss_threshold']>position['stop_loss_level']):
            position['stop_loss_level'] = round(position['max_profit_price']-position['stop_loss_threshold'], 6)
    
    if (position.get('side')=='SELL'):
        if (position['current_price']<position['max_profit_price']):
            position['max_profit_price'] = position['current_price']
        if (position['max_profit_price']+position['stop_loss_threshold']<position['stop_loss_level']):
            position['stop_loss_level'] = round(position['max_profit_price']+position['stop_loss_threshold'], 6)
            
    return position   
    
def update_take_profit_level(position):
    if(position['side']=='BUY'):
        position['take_profit_level'] = position['take_profit_level']+(.01*float(position['avgPrice']))
    if(position['side']=='SELL'):
        position['take_profit_level'] = position['take_profit_level']-(.01*float(position['avgPrice']))
    
    return position

def write_trade_to_database(open_position, close_position):
    if (open_position is None or close_position is None):
        return
    
    data_entry = {}
    data_entry['symbol'] = open_position.get('symbol')
    print(data_entry['symbol'])
    
    data_entry['open_trade_timestamp'] = datetime.datetime.utcfromtimestamp(int(open_position.get('time'))/1000)
    data_entry['open_trade_side'] = open_position.get('side')
    print('Open trade side:      ', data_entry['open_trade_side'])  
    data_entry['open_trade_price'] = float(open_position.get('avgPrice'))
    print('Open trade price:     ', data_entry['open_trade_price'])  
    data_entry['open_trade_quantity'] = float(open_position.get('executedQty'))
    data_entry['open_trade_position'] = float(open_position.get('cumQuote'))
    print('Open trade notional:  ', data_entry['open_trade_position'])
    
    data_entry['close_trade_timestamp'] = datetime.datetime.utcfromtimestamp(int(close_position.get('time'))/1000)  
    data_entry['close_trade_side'] = close_position.get('side')
    print('Close trade side:     ', data_entry['close_trade_side'])
    data_entry['close_trade_price'] = float(close_position.get('avgPrice'))
    print('CLose trade price:    ',data_entry['close_trade_price'])
    data_entry['close_trade_quantity'] = float(close_position.get('executedQty')) 
    data_entry['close_trade_position'] = float(close_position.get('cumQuote'))
    print('Close trade notional: ', data_entry['close_trade_position'])
        
    data_entry['trade_strategy'] = 'EMASAR Beach Entry 3m, 7/25 SMA/LMA'
    data_entry['transaction_costs'] = (data_entry['open_trade_position']+data_entry['close_trade_position'])*.0002
    data_entry['holding_period'] = str((data_entry['close_trade_timestamp'] - data_entry['open_trade_timestamp']))
    print('Holding Period:       ', data_entry['holding_period'])
        
    data_entry['percent_return'] = math.nan    
    data_entry['dollar_PnL'] = math.nan
    
    if (data_entry['open_trade_side'] == 'BUY'):
        data_entry['percent_return'] = ((data_entry['close_trade_price']/data_entry['open_trade_price'])-1)*100
        data_entry['dollar_PnL']     = (data_entry['close_trade_price']-data_entry['open_trade_price'])*data_entry['close_trade_quantity']
    if (data_entry['open_trade_side'] == 'SELL'):
        data_entry['percent_return'] = ((data_entry['open_trade_price']/data_entry['close_trade_price'])-1)*100
        data_entry['dollar_PnL']     = -1*((data_entry['close_trade_price']-data_entry['open_trade_price'])*data_entry['close_trade_quantity'])
    
    print('Percent return:       ', data_entry['percent_return'])
    print('Dollar PnL:           ', data_entry['dollar_PnL'])
    print()
    rows = []
    df = pd.DataFrame()
    rows.append(data_entry)
    df = df.append(rows)
    con = sqlite3.connect('all_trades_v2-4.db', timeout=1200) 
    
    df.to_sql('trade_log', con, index = False, if_exists='append')
    con.commit()
 
def write_open_positions_to_database(all_positions):
    con = sqlite3.connect('open_positions_beach_entry_3m_7_25_v2.db', timeout=1200) 
    con.execute('''DROP TABLE positions''')
    con.execute('''CREATE TABLE IF NOT EXISTS positions (
                    orderId             text, 
                    symbol              text, 
                    status              text,
                    price               text,
                    avgPrice            text,
                    origQty             text, 
                    executedQty         text, 
                    cumQuote            text, 
                    timeInForce         text, 
                    type                text, 
                    side                text, 
                    time                text, 
                    stop_loss_threshold REAL, 
                    stop_loss_level     REAL, 
                    take_profit_level   REAL,
                    current_price       REAL, 
                    max_profit_price    REAL)''')

    for pos in all_positions:
        data_entry = {}
        data_entry['orderId']               = pos['orderId']
        data_entry['symbol']                = pos['symbol']
        data_entry['status']                = pos['status']
        data_entry['price']                 = pos['price']
        data_entry['avgPrice']              = pos['avgPrice']
        data_entry['origQty']               = pos['origQty']
        data_entry['executedQty']           = pos['executedQty']
        data_entry['cumQuote']              = pos['cumQuote']
        data_entry['timeInForce']           = pos['timeInForce']
        data_entry['type']                  = pos['type']
        data_entry['side']                  = pos['side']
        data_entry['time']                  = pos['time']
        data_entry['stop_loss_threshold']   = pos['stop_loss_threshold']
        data_entry['stop_loss_level']       = pos['stop_loss_level']
        data_entry['take_profit_level']     = pos['take_profit_level']
        data_entry['current_price']         = pos['current_price']
        data_entry['max_profit_price']      = pos['max_profit_price']
        
        rows = []
        rows.append(data_entry)
        df = pd.DataFrame()
        df = df.append(rows)
        con = sqlite3.connect('open_positions_beach_entry_3m_7_25_v2.db', timeout=1200) 
        
        df.to_sql('positions', con, index = False, if_exists='append')
        con.commit()
        
def create_open_positions_database():
    con = sqlite3.connect('open_positions_beach_entry_3m_7_25_v2.db', timeout=1200) 
    con.execute('''CREATE TABLE IF NOT EXISTS positions (
                    orderId             text, 
                    symbol              text, 
                    status              text,
                    price               text,
                    avgPrice            text,
                    origQty             text, 
                    executedQty         text, 
                    cumQuote            text, 
                    timeInForce         text, 
                    type                text, 
                    side                text, 
                    time                text, 
                    stop_loss_threshold REAL, 
                    stop_loss_level     REAL, 
                    take_profit_level   REAL,
                    current_price       REAL, 
                    max_profit_price    REAL, 
                    take_profit_price   REAL)''')
    
def read_open_positions_from_database():
    all_positions = []
    db_file = 'open_positions_beach_entry_3m_7_25_v2.db'
    conn = sqlite3.connect(db_file, timeout=1200)
    df = pd.read_sql_query('SELECT * FROM positions', con = conn)
    all_positions = df.to_dict('records')
    
    return all_positions 

def print_positions_to_console(all_positions):
    # Print all positions into console
    print('Account Balance:   ', round(float(client.futures_account_balance()[0].get('balance')), 2))
    print('Unrealized Profit: ', round(float(client.futures_account()['totalUnrealizedProfit']), 2))  
    print('Margin Remaining:  ', round(float(client.futures_account_balance()[0].get('withdrawAvailable')), 2))
   
    format_for_console = '{:<12} {:<5} {:<16} {:<12} {:<15} {:<12} {:<12} {:<12} {:<12} {:<9} {:<9} {:<9} {:<10} {:<14}'
    print(format_for_console.format(
            'symbol',
            'side',
            'holding_period',
            'entryPrice',
            'stop_threshold',
            'stop_loss',
            'currentPrice',
            'take_profit',
            'max_return',
            'return',
            'proj',
            'diff',
            'threshold',
            'position_size'))
   
    average = {'num_positions':0.0,
               'avg_holding_period': datetime.timedelta(hours=0),
               'avg_max_return':0.0,
               'avg_return':0.0,
               'avg_proj':0.0,
               'avg_diff':0.0,
               'avg_threshold':0.0, 
               'total_position_size': 0.0}
   
    for pos in all_positions:
        if(pos['side']=='BUY'):
            sign=1.0
        else:
            sign=-1.0
           
        print (format_for_console.format(
                pos['symbol'],
                pos['side'],
                str((datetime.datetime.now() -
                     datetime.datetime.utcfromtimestamp(int(pos['time'])/1000) +
                     datetime.timedelta(hours = 5))),
                pos['avgPrice'],
                pos['stop_loss_threshold'],
                pos['stop_loss_level'],
                pos['current_price'],
                sign*round((((pos['take_profit_level']/float(pos['avgPrice']))-1)*10000), 0),
                sign*round((((pos['max_profit_price']/float(pos['avgPrice']))-1)*10000), 0),
                sign*round((((pos['current_price']/float(pos['avgPrice']))-1)*10000), 0),
                sign*round((((pos['stop_loss_level']/float(pos['avgPrice']))-1)*10000), 0),
                sign*(round((((pos['current_price']/float(pos['avgPrice']))-1)*10000), 0)-
                      round((((pos['stop_loss_level']/float(pos['avgPrice']))-1)*10000), 0)),
                round(((pos['stop_loss_threshold']/float(pos['avgPrice']))*10000), 0),
                round((float(pos['avgPrice'])*float(pos['executedQty'])), 1)))
       
        average['num_positions']      = average['num_positions']+1
        average['avg_holding_period'] = average['avg_holding_period']+(datetime.datetime.now() -
                                                                       datetime.datetime.utcfromtimestamp(int(pos['time'])/1000) +
                                                                       datetime.timedelta(hours = 5))
        average['avg_max_return']= average['avg_max_return']+sign*(((pos['max_profit_price']/float(pos['avgPrice']))-1)*10000)
        average['avg_return']    = average['avg_return']    +sign*(((pos['current_price']/float(pos['avgPrice']))-1)*10000)
        average['avg_proj']      = average['avg_proj']      +sign*(((pos['stop_loss_level']/float(pos['avgPrice']))-1)*10000)
        average['avg_diff']      = average['avg_diff']      +sign*((((pos['current_price']/float(pos['avgPrice']))-1)*10000) -
                                                                   (((pos['stop_loss_level']/float(pos['avgPrice']))-1)*10000))
        average['avg_threshold'] = average['avg_threshold'] +((pos['stop_loss_threshold']/float(pos['avgPrice']))*10000)
        average['total_position_size'] = average['total_position_size']+round((float(pos['avgPrice'])*float(pos['executedQty'])), 1)
        
    if (average['num_positions']>0):
        print (format_for_console.format(
                'AVERAGE',
                '',
                str(average['avg_holding_period']/average['num_positions']),
                '',
                '',
                '',
                '',
                '',
                round(average['avg_max_return']/average['num_positions'], 0),
                round(average['avg_return']/average['num_positions'], 0),
                round(average['avg_proj']/average['num_positions'], 0),
                round(average['avg_diff']/average['num_positions'], 0),
                round(average['avg_threshold']/average['num_positions'], 0),
                round(average['total_position_size'])))

def main():      
    rand_num = 1
    all_positions = []
    create_open_positions_database()
    # This loop is executed once every 20 seconds. 
    #    First check to see if position crosses stop-loss or take profit levels.
    #       If so, close positions. 
    #    Second check for new emails. If email exists, process information and
    #       open position if no position exists. 
    # All current positions are managed by all_positions list. When a position is  
    #   active, the order information is stored in all_positions. 
    
    while (rand_num<500000):   
        print('Alert # '+str(rand_num)+" "+str(datetime.datetime.now()))
        
        # Get all positions from locally stored database
        all_positions = read_open_positions_from_database()
        
        # If exception is thrown it is likely due to internet connectivity 
        #   Give laptop 60 seconds to reestablish internet connection. 
        try:
            alerts = readmail()  
        except Exception as e:
            time.sleep(5.0)
        
        # Check to see if position has fallen below stop-loss level on long  positions
        #                          has risen  above stop-loss level on short positions
        # If position has crossed stop-loss level, close position. 
        for pos in all_positions:
            pos = update_stop_loss_level(pos)
        
        temp = []
        for pos in all_positions:
            # Check to see if current price is below stop-loss level for long positions. 
            #   If so, close position and do not append close position to all_positions list
            if ((pos.get('side')=='BUY') and 
                (pos['current_price']<=pos['stop_loss_level'])):
                    close_position = make_closing_trade(symbol        = pos.get('symbol'),
                                                        long_or_short = 'Short', 
                                                        quantity      = pos.get('executedQty'))
                    write_trade_to_database(open_position = pos, 
                                            close_position = close_position)
                    
            # Check to see if current price is above stop-loss level for short positions. 
            #   If so, close position and do not append close position to all_positions list
            elif ((pos.get('side')=='SELL') and 
                  (pos['current_price']>=pos['stop_loss_level'])):
                    close_position = make_closing_trade(symbol        = pos.get('symbol'),
                                                        long_or_short = 'Long', 
                                                        quantity      = pos.get('executedQty'))
                    write_trade_to_database(open_position = pos, 
                                            close_position = close_position)
                    
            # If current price does not cross stop-loss level, add pos to temp
            else:
                temp.append(pos)
        all_positions = temp
        
        # Check to see if position has risen above  take-profit level on long  positions
        #                          has fallen below take-profit level on short positions
        # If position has crossed take-profit, trim position 10%
        for pos in all_positions:
            if ((pos['side']=='BUY') and 
                (pos['current_price']>=pos['take_profit_level'])):
                    close_position = make_closing_trade(symbol        = pos['symbol'],
                                                        long_or_short = 'Short', 
                                                        quantity      = round(float(client.futures_position_information(symbol = pos['symbol'])[0]['positionAmt'])*.15, get_significance_level(symbol = pos['symbol'])))
                    if (close_position is not None):
                        write_trade_to_database(open_position = pos, 
                                                close_position = close_position)
                    pos = update_take_profit_level(pos)
                    pos['executedQty'] = float(client.futures_position_information(symbol = pos['symbol'])[0]['positionAmt'])
        
            if ((pos['side']=='SELL') and 
                  (pos['current_price']<=pos['take_profit_level'])):
                    close_position = make_closing_trade(symbol        = pos['symbol'],
                                                        long_or_short = 'Long', 
                                                        quantity      = round(-1*float(client.futures_position_information(symbol = pos['symbol'])[0]['positionAmt'])*.15, get_significance_level(symbol = pos['symbol'])))
                    if (close_position is not None):
                        write_trade_to_database(open_position = pos, 
                                                close_position = close_position)
                        
                    pos = update_take_profit_level(pos)
                    pos['executedQty'] = -1*float(client.futures_position_information(symbol = pos['symbol'])[0]['positionAmt'])
        
        
        # If there is no new email to be read, alerts will be empty. 
        # If there are  new emails to be read, process email info below. 
        for a in alerts:
            try:    
                symbol = a[1]+"USDT"
            except IndexError: continue 
            ### OPENING TRADE CHECK FOR POTENTIAL LONG POSITION
            if (a[0] == 'Alert: BEACH CROSSING UP'):
                # Check to see if Position already exists
                position_exists = False
                for pos in all_positions:
                    if (pos is not None):
                        if (pos.get('symbol') == symbol):
                            position_exists = True
                # If position does not exist, create long position. 
                if (position_exists == False):  
                    try:
                        beach_l1 = float(a[20].split('=', 1)[1])
                    except Exception as e:
                        beach_l1 = None
                    try:
                        beach_l2 = float(a[19].split('=', 1)[1])
                    except Exception as e:
                        beach_l2 = None
                    try:
                        sky_l2 = float(a[23].split('=', 1)[1])
                    except Exception as e:
                        sky_l2 = None
        
        
                    con = sqlite3.connect('all_trades.db', timeout=1200) 
                    trade_history = pd.read_sql_query('SELECT * FROM trade_log', con = con)
                    trade_history = trade_history[trade_history.symbol == symbol]
                    trade_history = trade_history.sort_values(by='close_trade_timestamp', ascending=False)
                    if (not trade_history.empty):
                        previous_trade_side = trade_history.iloc[0]['open_trade_side']
                        previous_trade_pnl = trade_history.iloc[0]['percent_return']
                    else:
                        previous_trade_side = None
                        previous_trade_pnl = None
                    
                    if (beach_l1 != None and beach_l2 != None and previous_trade_side != None and previous_trade_pnl != None):
                        if (beach_l1>beach_l2 and not (previous_trade_side=='BUY' and previous_trade_pnl<-1.5)):
                            
                            #If previous position was a long position and closed out at more than 1$ loss
                            # do not make trade
                            new_position = make_opening_trade(symbol = symbol, 
                                                              long_or_short = 'Long')
                            if (new_position is not None):
                                    new_position = add_stop_loss_level(new_position)
                                    new_position = add_take_profit_level(position = new_position, 
                                                                         take_profit = sky_l2)
                                    all_positions.append(new_position)
            
            ### OPENING TRADE CHECK FOR POTENTIAL SHORT POSITION
            if (a[0] == 'Alert: BEACH CROSSING DOWN'):
                # Check to see if Position already exists
                position_exists = False
                for pos in all_positions:
                    if (pos is not None):
                        if (pos.get('symbol') == symbol):
                            position_exists = True
                # If position does not exist, create position. 
                if (position_exists == False):  
                    try:
                        beach_l1 = float(a[20].split('=', 1)[1])
                    except Exception as e:
                        beach_l1 = None
                    try:
                        beach_l2 = float(a[19].split('=', 1)[1])
                    except Exception as e:
                        beach_l2 = None
                    try:
                        sky_l2 = float(a[23].split('=', 1)[1])
                    except Exception as e:
                        sky_l2 = None
                    
                    con = sqlite3.connect('all_trades.db', timeout=1200) 
                    trade_history = pd.read_sql_query('SELECT * FROM trade_log', con = con)
                    trade_history = trade_history[trade_history.symbol == symbol]
                    trade_history = trade_history.sort_values(by='close_trade_timestamp', ascending=False)
                    if (not trade_history.empty):
                        previous_trade_side = trade_history.iloc[0]['open_trade_side']
                        previous_trade_pnl = trade_history.iloc[0]['percent_return']
                    else:
                        previous_trade_side = None
                        previous_trade_pnl = None
                    
                    if (beach_l1 != None and beach_l2 != None and previous_trade_side != None and previous_trade_pnl != None):
                        if (beach_l1<beach_l2 and not (previous_trade_side=='SELL' and previous_trade_pnl<-1.5)):
                            new_position = make_opening_trade(symbol = symbol, 
                                                              long_or_short = 'Short')
                            if (new_position is not None):
                                new_position = add_stop_loss_level(new_position)
                                new_position = add_take_profit_level(position = new_position, 
                                                                         take_profit = sky_l2)
                                all_positions.append(new_position)
            new_position = {} 
        
        
        print_positions_to_console(all_positions)
        
        # Store open positions to local saved database
        write_open_positions_to_database(all_positions)
        
        # Close any positions that for some reason were not copied to local database
        print('# of Hanging Positions: ', close_hanging_positions(all_positions = all_positions))
        print()
        
        rand_num = rand_num + 1
        time.sleep(10.0)
        
