import yfinance as yf
import ta  # Added import for ta library
import csv
import numpy as np
from nsepython import *
import pandas as pd
import warnings
import streamlit as st
from datetime import datetime
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables
now = datetime.now()
date = now.strftime("%Y-%m-%d %H-%M-%S")
names = ['STOCK', 'RSI', 'PRICE VS MA50', 'MACD', 'VWAP', 'NIFTYTREND', 'BOLLINGERBANDS', 
         'ATR STOP-LOSS', 'BULLISH INDICATOR COUNT', 'RECOMMENDATION', 'PROFIT POTENTIAL']

def check_moving_average_conditions(stock_ticker):
    try:
        # Fetch stock data
        data = yf.download(stock_ticker, period="59d", interval="15m")

        # Debug: Print column names
        

        # Fix MultiIndex issue by flattening column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in data.columns]

        # Debug: Print fixed column names


        # Correct column name for 'Close'
        close_col = f'Close_{stock_ticker}'

        # Ensure column exists
        if close_col not in data.columns:
            return None, f"No data found for {stock_ticker}. Check the ticker symbol."

        # Extract close prices
        close_prices = data[close_col].dropna()

        # Ensure we have enough data
        if len(close_prices) < 50:
            return None, f"Not enough data to calculate moving averages for {stock_ticker}."

        # Compute moving averages
        ma_10 = ta.trend.sma_indicator(close_prices, window=10)
        ma_20 = ta.trend.sma_indicator(close_prices, window=20)
        ma_50 = ta.trend.sma_indicator(close_prices, window=50)

        # Get latest values
        latest_price = close_prices.iloc[-1]
        latest_ma_10 = ma_10.iloc[-1]
        latest_ma_20 = ma_20.iloc[-1]
        latest_ma_50 = ma_50.iloc[-1]

        # Define conditions
        price_above_50 = latest_price > latest_ma_50
        near_10_20 = (
            abs(latest_price - latest_ma_10) < (latest_price * 0.075) and
            abs(latest_price - latest_ma_20) < (latest_price * 0.075)
        )

        # Store results
        L2 = [latest_price, latest_ma_10, latest_ma_20, latest_ma_50]

        if price_above_50 and near_10_20:
            return L2, f"{stock_ticker}: ✅ Conditions Met (Price: {latest_price:.2f}, MA10: {latest_ma_10:.2f}, MA20: {latest_ma_20:.2f}, MA50: {latest_ma_50:.2f})"
        else:
            return False, f"{stock_ticker}: ❌ Conditions NOT Met (Price: {latest_price:.2f}, MA10: {latest_ma_10:.2f}, MA20: {latest_ma_20:.2f}, MA50: {latest_ma_50:.2f})"

    except Exception as e:
        return None, f"Error in moving average conditions for {stock_ticker}: {e}"

# RSI Calculation (Optional: Replace manual with ta.momentum.rsi)
def calculate_rsi_manual(stock_ticker):
    stock_data = yf.download(stock_ticker, period='59d', interval='15m')

    # Extract closing prices and ensure it is a Pandas Series
    close_prices = stock_data['Close'].squeeze()  

    # Calculate RSI
    rsi = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi()
    
    # Print only the latest RSI value
    A=(round(rsi.dropna().iloc[-1], 2))
    return A # Rounds to 2 decimal places


# MACD Crossover
def check_macd_crossover(data):
    close = data['Close'].astype(np.float64)  # Keep as pandas Series
    if len(close) < 34:
        return "Insufficient data for MACD"
    try:
        # Replace talib.MACD with ta.trend.MACD
        macd_indicator = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
        macd = macd_indicator.macd()
        signal = macd_indicator.macd_signal()
        return ("Bullish Crossover" if macd.iloc[-1] > signal.iloc[-1] else 
                "Bearish Crossover" if macd.iloc[-1] < signal.iloc[-1] else "Neutral")
    except Exception as e:
        return f"MACD failed: {e}"

# ATR Stop Loss and Take Profit
def calculate_atr_stop_loss(data, entry_price, atr_multiplier=1.5, atr_period=14):
    high = data['High'].astype(np.float64)
    low = data['Low'].astype(np.float64)
    close = data['Close'].astype(np.float64)
    if len(close) < atr_period:
        return entry_price - 5, entry_price + 5
    try:
        # Replace talib.ATR with ta.volatility.AverageTrueRange
        atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=atr_period)
        atr = atr_indicator.average_true_range().iloc[-1]
        return entry_price - (atr * atr_multiplier), entry_price + (atr * 2)
    except Exception as e:
        return entry_price - 5, entry_price + 5

# VWAP Check (Remains manual as ta VWAP requires specific setup)
def check_vwap(data, current_date):
    today_data = data[data.index.date == current_date].copy()
    if len(today_data) < 2:
        return "No intraday data available"
    today_data['TP'] = (today_data['High'] + today_data['Low'] + today_data['Close']) / 3
    today_data['VWAP'] = (today_data['TP'] * today_data['Volume']).cumsum() / today_data['Volume'].cumsum()
    return "Above VWAP (Bullish)" if today_data['Close'].iloc[-1] > today_data['VWAP'].iloc[-1] else "Below VWAP (Bearish)"

# Nifty Trend
def check_nifty_trend():
    try:
        nifty = yf.download("^NSEI", period="60d", interval="15m", auto_adjust=True)
        if nifty.empty:
            nifty = yf.download("^NIFTY50", period="60d", interval="15m", auto_adjust=True)
        if nifty.empty:
            return "No sufficient Nifty data"
        
        close = nifty['Close'].dropna().values.astype(np.float64).flatten()
        if len(close) < 50:
            return "Insufficient data for SMA calculation"
        
        df = pd.DataFrame({'Close': close})
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        
        if pd.isna(df['SMA_50'].iloc[-1]):
            return "SMA calculation failed due to missing data"
        
        return "Nifty Above MA50 (Bullish)" if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1] else "Nifty Below MA50 (Bearish)"
    except Exception as e:
        return f"Error fetching Nifty trend: {e}"

# Bollinger Bands
def check_bollinger_bands(data):
    close = data['Close'].astype(np.float64)  # Keep as pandas Series
    if len(close) < 20:
        return "Insufficient data for Bollinger Bands"
    try:
        # Replace talib.BBANDS with ta.volatility.BollingerBands
        bb_indicator = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        upper = bb_indicator.bollinger_hband()
        middle = bb_indicator.bollinger_mavg()
        lower = bb_indicator.bollinger_lband()
        latest_price = close.iloc[-1]
        if latest_price <= lower.iloc[-1]:
            return "Near Lower Band (Potential Buy)"
        elif latest_price >= upper.iloc[-1]:
            return "Near Upper Band (Potential Sell)"
        return "Inside Bands"
    except Exception as e:
        return f"Bollinger Bands failed: {e}"

# Stock Analysis
def analyze_stock(stock_ticker):
    try:
        ticker = stock_ticker if stock_ticker.endswith('.NS') else stock_ticker + '.NS'
        data = yf.download(ticker, period="60d", interval="15m", auto_adjust=False)
        if data.empty:
            return None, f"No data found for {ticker}"

        if data.columns.nlevels > 1:
            data.columns = [col[0] for col in data.columns]

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns) or len(data) < 50:
            return None, f"Insufficient or invalid data for {ticker} (Rows: {len(data)})"

        latest_price = data['Close'].iloc[-1]
        close_array = data['Close'].to_numpy().astype(np.float64)  # For manual RSI; could use Series

        if not np.issubdtype(close_array.dtype, np.number):
            return None, "Error: Close array contains non-numeric data"
        if np.any(~np.isfinite(close_array)):
            close_array = pd.Series(close_array).fillna(method='ffill').fillna(method='bfill').to_numpy()
        close_array = close_array.astype(np.float64)

        # Optionally replace manual RSI with ta.momentum.rsi
        rsi = calculate_rsi_manual(stock_ticker)
        # Alternative: rsi = ta.momentum.rsi(data['Close'], window=14).iloc[-1] or 50 if NaN
        rsi_bullish = rsi > 52

        try:
            # Replace talib.SMA with ta.trend.sma_indicator
            ma_50 = ta.trend.sma_indicator(data['Close'], window=50).iloc[-1]
        except Exception as e:
            ma_50 = data['Close'].iloc[-50:].mean()
        price_above_ma50 = latest_price > ma_50

        macd_status = check_macd_crossover(data)
        vwap_status = check_vwap(data, data.index[-1].date())
        nifty_status = check_nifty_trend()
        bb_status = check_bollinger_bands(data)
        stop_loss, take_profit = calculate_atr_stop_loss(data, latest_price)

        macd_bullish = macd_status == "Bullish Crossover"
        vwap_bullish = vwap_status == "Above VWAP (Bullish)"
        nifty_bullish = nifty_status == "Nifty Above MA50 (Bullish)"
        bb_bullish = bb_status in ["Near Lower Band (Potential Buy)", "Inside Bands"]
        profit_potential = (take_profit - latest_price) >= 4
        profit = (take_profit - latest_price)

        bullish_count = sum([rsi_bullish, price_above_ma50, macd_bullish, vwap_bullish, nifty_bullish, bb_bullish])
        recommendation = f"Buy at {latest_price:.2f}, Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}" if bullish_count >= 4 and profit_potential else "Do not buy"

        stock_data = [
            stock_ticker,
            f"{rsi:.2f} ({'Bullish' if rsi_bullish else 'Not Bullish'})",
            'Above' if price_above_ma50 else 'Below',
            macd_status,
            vwap_status,
            nifty_status,
            bb_status,
            f"{stop_loss:.2f}, {take_profit:.2f}",
            f"{bullish_count}",
            recommendation,
            profit
        ]

        output = f"""
Stock: {stock_ticker}
RSI: {rsi:.2f} ({'Bullish' if rsi_bullish else 'Not Bullish'})
Price vs MA50: {'Above' if price_above_ma50 else 'Below'})
MACD: {macd_status}
VWAP: {vwap_status}
Nifty Trend: {nifty_status}
Bollinger Bands: {bb_status}
ATR Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}
Bullish Indicators Count: {bullish_count}/6
Recommendation: {recommendation}
Profit Potential: {profit}
"""
        return stock_data, output
    except Exception as e:
        return None, f"Error analyzing stock: {e}"

# Modified Email Sending Function (unchanged)
def send_email(subject, body, csv_data, recipient_email, sender_email, sender_password):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(csv_data)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="analysis.csv"')
        msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {e}"

# Streamlit App (unchanged except for function calls using ta)
def main():
    st.title("Stock Analysis Dashboard")

    option = st.sidebar.selectbox("Choose an Option", ["Specific Stock", "Latest Gainers", "Intraday Stocks"])

    if option == "Specific Stock":
        st.header("Analyze a Specific Stock")
        ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE)")
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                stock_data, output = analyze_stock(ticker)
                st.write(output)
                if stock_data and not isinstance(stock_data, str):
                    df = pd.DataFrame([stock_data], columns=names)
                    st.dataframe(df)
                    csv_content = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name="specific_stock_analysis.csv",
                        mime="text/csv"
                    )
                    if st.checkbox("Send Email"):
                        recipient = st.text_input("Recipient Email")
                        sender = st.text_input("Your Email")
                        password = st.text_input("App Password", type="password")
                        if st.button("Send"):
                            subject = f"Stock Analysis Report for {ticker} - {date}"
                            body = f"Attached is the analysis for {ticker} generated on {date}.\n\n{output}"
                            success, msg = send_email(subject, body, csv_content, recipient, sender, password)
                            st.write(msg)

    elif option == "Latest Gainers":
        st.header("Analyze Latest Gainers")
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")

        if uploaded_file and st.button("Analyze"):
            decoded_file = uploaded_file.getvalue().decode("utf-8").splitlines()
            reader = csv.reader(decoded_file)
            all_data = list(reader)

            header = []
            for row in all_data[:20]:
                for item in row:
                    clean_item = item.replace('"', '').replace('\ufeff', '').strip()
                    if clean_item.startswith(","):
                        clean_item = clean_item[1:].strip()
                    if clean_item:
                        header.append(clean_item)

            stock_data1 = all_data[4:]
            try:
                symbol_idx = header.index("SYMBOL")
                chng_idx = header.index("CHNG")
            except ValueError:
                st.error("CSV must contain 'SYMBOL' and 'CHNG' columns.")
                st.stop()

            stock_changes = []
            for row in stock_data1:
                if len(row) > max(symbol_idx, chng_idx):
                    stock_name = row[symbol_idx].strip()
                    try:
                        change_value = float(row[chng_idx].strip())
                    except ValueError:
                        st.warning(f"Skipping {stock_name}: Invalid CHNG value '{row[chng_idx]}'")
                        continue
                    stock_changes.append((stock_name, change_value))

            filtered_stocks = [stock for stock, change in stock_changes if change >= 0]
            total = len(filtered_stocks)
            progress_bar = st.progress(0)
            results = []

            for i, stock in enumerate(filtered_stocks, 1):
                stock_ticker = stock + '.NS'
                result, msg = check_moving_average_conditions(stock_ticker)
                st.write(msg)
                if result != False:
                    stock_data, output = analyze_stock(stock_ticker)
                    st.write(output)
                    if stock_data and not isinstance(stock_data, str):
                        results.append(stock_data)
                progress_bar.progress(i / total)

            if results:
                df_results = pd.DataFrame(results, columns=names)
                st.dataframe(df_results)
                csv_content = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv_content,
                    file_name="gainerstock.csv",
                    mime="text/csv"
                )

                if st.checkbox("Send Email"):
                    recipient = st.text_input("Recipient Email")
                    sender = st.text_input("Your Email")
                    password = st.text_input("App Password", type="password")
                    if st.button("Send"):
                        subject = f"Latest Gainers Analysis Report - {date}"
                        body = f"Attached is the gainers analysis report generated on {date}."
                        success, msg = send_email(subject, body, csv_content, recipient, sender, password)
                        st.write(msg)

    elif option == "Intraday Stocks":
        st.header("Analyze Intraday Stocks")
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                intraday_stocks = ['20MICRONS.NS', '5PAISA.NS', 'AADHARHFC.NS', 'AARTIDRUGS.NS', 'AARTIIND.NS', 'ABDL.NS',
 'ACI.NS', 'ACL.NS', 'ACLGATI.NS', 'ACMESOLAR.NS', 'ADFFOODS.NS', 'ADSL.NS', 'ADVANIHOTR.NS',
 'ADVENZYMES.NS', 'AEROFLEX.NS', 'AFCONS.NS', 'AHL.NS', 'AKUMS.NS', 'ALEMBICLTD.NS', 'ALLCARGO.NS',
 'ALOKINDS.NS', 'AMNPLST.NS', 'ANDHRAPAP.NS', 'ANDHRSUGAR.NS', 'APCOTEXIND.NS', 'APEX.NS', 'APOLLO.NS',
 'APOLLOPIPE.NS', 'APTECHT.NS', 'APTUS.NS', 'ARIHANTCAP.NS', 'ARIHANTSUP.NS', 'ARKADE.NS', 'ARVIND.NS',
 'ARVINDFASN.NS', 'ASHAPURMIN.NS', 'ASHIANA.NS', 'ASHOKA.NS', 'ASIANENE.NS', 'ASIANHOTNR.NS', 'ASIANTILES.NS',
 'ASKAUTOLTD.NS', 'ATL.NS', 'ATULAUTO.NS', 'AURUM.NS', 'AVADHSUGAR.NS', 'AVANTEL.NS', 'AVL.NS', 'AVTNPL.NS', 'AYMSYNTEX.NS',
 'BAJAJCON.NS', 'BAJAJHIND.NS', 'BAJEL.NS', 'BALAJITELE.NS', 'BALMLAWRIE.NS', 'BALRAMCHIN.NS', 'BANCOINDIA.NS',
 'BANSALWIRE.NS', 'BARBEQUE.NS', 'BBOX.NS', 'BCLIND.NS', 'BEPL.NS', 'BESTAGRO.NS', 'BFINVEST.NS', 'BGRENERGY.NS',
 'BHAGCHEM.NS', 'BHAGERIA.NS', 'BHARATWIRE.NS', 'BIGBLOC.NS', 'BIRLAMONEY.NS', 'BLACKBUCK.NS', 'BLAL.NS', 'BLISSGVS.NS',
 'BLKASHYAP.NS', 'BLS.NS', 'BLSE.NS', 'BODALCHEM.NS', 'BOMDYEING.NS', 'BOROLTD.NS', 'BOROSCI.NS', 'BSHSL.NS', 'BSOFT.NS',
 'CAMLINFINE.NS', 'CAMPUS.NS', 'CANTABIL.NS', 'CAPACITE.NS', 'CAPITALSFB.NS', 'CAREERP.NS', 'CCCL.NS', 'CEIGALL.NS',
 'CENTRUM.NS', 'CESC.NS', 'CGCL.NS', 'CHEMCON.NS', 'CHEMPLASTS.NS', 'CHOICEIN.NS', 'CIEINDIA.NS', 'CLSEL.NS', 'CMSINFO.NS',
 'CONFIPET.NS', 'CONSOFINVT.NS', 'CREST.NS', 'CSBBANK.NS', 'CSLFINANCE.NS', 'CUB.NS', 'CUPID.NS', 'CYBERTECH.NS', 'CYIENTDLM.NS',
 'DALMIASUG.NS', 'DBCORP.NS', 'DBEIL.NS', 'DBL.NS', 'DBREALTY.NS', 'DCAL.NS', 'DCBBANK.NS', 'DCMSRIND.NS', 'DCW.NS',
 'DCXINDIA.NS', 'DEEDEV.NS', 'DEEPINDS.NS', 'DELTACORP.NS', 'DEN.NS', 'DHAMPURSUG.NS', 'DHANBANK.NS', 'DHANI.NS', 'DHARMAJ.NS',
 'DIACABS.NS', 'DIFFNKG.NS', 'DIGISPICE.NS', 'DISHTV.NS', 'DLINKINDIA.NS', 'DMCC.NS', 'DOLATALGO.NS', 'DOLLAR.NS', 'DOLPHIN.NS',
 'DONEAR.NS', 'DPSCLTD.NS', 'DREAMFOLKS.NS', 'DVL.NS', 'DWARKESH.NS', 'EASEMYTRIP.NS', 'ECOSMOBLTY.NS', 'EDELWEISS.NS',
 'EIEL.NS', 'EIHAHOTELS.NS', 'EKC.NS', 'ELECON.NS', 'ELECTCAST.NS', 'ELGIEQUIP.NS', 'ELIN.NS', 'EMAMIPAP.NS', 'EMBDL.NS',
 'EMIL.NS', 'EMKAY.NS', 'ENGINERSIN.NS', 'ENIL.NS', 'EPACK.NS', 'EPL.NS', 'EQUITASBNK.NS', 'ESAFSFB.NS', 'ESSARSHPNG.NS',
 'ESTER.NS', 'EVEREADY.NS', 'EXICOM.NS', 'FAZE3Q.NS', 'FCL.NS', 'FDC.NS', 'FEDFINA.NS', 'FILATEX.NS', 'FINOPB.NS',
 'FINPIPE.NS', 'FIRSTCRY.NS', 'FLAIR.NS', 'FMGOETZE.NS', 'FOCUS.NS', 'FOODSIN.NS', 'FUSION.NS', 'GABRIEL.NS',
 'GAEL.NS', 'GALLANTT.NS', 'GANDHAR.NS', 'GANESHBE.NS', 'GARUDA.NS', 'GATEWAY.NS', 'GEECEE.NS', 'GENUSPOWER.NS',
 'GEOJITFSL.NS', 'GEPIL.NS', 'GFLLIMITED.NS', 'GHCLTEXTIL.NS', 'GICHSGFIN.NS', 'GIPCL.NS', 'GMDCLTD.NS',
 'GMRP&UI.NS', 'GNA.NS', 'GOCLCORP.NS', 'GODAVARIB.NS', 'GOKULAGRO.NS', 'GOLDIAM.NS', 'GOPAL.NS', 'GPIL.NS',
 'GPPL.NS', 'GPTHEALTH.NS', 'GPTINFRA.NS', 'GRAPHITE.NS', 'GREAVESCOT.NS', 'GREENPANEL.NS', 'GREENPLY.NS', 'GREENPOWER.NS',
 'GRMOVER.NS', 'GSFC.NS', 'GSPL.NS', 'GTLINFRA.NS', 'GTPL.NS', 'GUFICBIO.NS', 'GULPOLY.NS', 'GVKPIL.NS', 'HARIOMPIPE.NS',
 'HARSHA.NS', 'HATHWAY.NS', 'HCC.NS', 'HCG.NS', 'HEG.NS', 'HEIDELBERG.NS', 'HEMIPROP.NS', 'HERANBA.NS', 'HERCULES.NS',
 'HERITGFOOD.NS', 'HEXATRADEX.NS', 'HFCL.NS', 'HIKAL.NS', 'HIMATSEIDE.NS', 'HINDCOMPOS.NS', 'HINDMOTORS.NS', 'HINDOILEXP.NS',
 'HINDWAREAP.NS', 'HITECH.NS', 'HLEGLAS.NS', 'HLVLTD.NS', 'HMAAGRO.NS', 'HMT.NS', 'HMVL.NS', 'HONASA.NS', 'HPL.NS',
 'HUBTOWN.NS', 'HUHTAMAKI.NS', 'ICIL.NS', 'IDEAFORGE.NS', 'IEX.NS', 'IFCI.NS', 'IFGLEXPOR.NS', 'IGL.NS', 'IGPL.NS',
 'IIFL.NS', 'IITL.NS', 'IKIO.NS', 'IMAGICAA.NS', 'INDIACEM.NS', 'INDIANHUME.NS', 'INDOAMIN.NS', 'INDOBORAX.NS',
 'INDOCO.NS', 'INDORAMA.NS', 'INDOSTAR.NS', 'INDRAMEDCO.NS', 'INDSWFTLAB.NS', 'INFIBEAM.NS', 'INFOBEAN.NS',
 'INNOVANA.NS', 'INOXGREEN.NS', 'IOLCP.NS', 'IPL.NS', 'IRCON.NS', 'IRIS.NS', 'IRMENERGY.NS', 'IXIGO.NS',
 'J&KBANK.NS', 'JAGRAN.NS', 'JAGSNPHARM.NS', 'JAIBALAJI.NS', 'JAICORPLTD.NS', 'JAMNAAUTO.NS', 'JAYAGROGN.NS',
 'JAYBARMARU.NS', 'JAYNECOIND.NS', 'JGCHEM.NS', 'JINDALSAW.NS', 'JINDWORLD.NS', 'JISLDVREQS.NS', 'JISLJALEQS.NS',
 'JITFINFRA.NS', 'JKPAPER.NS', 'JKTYRE.NS', 'JMFINANCIL.NS', 'JNKINDIA.NS', 'JPASSOCIAT.NS', 'JPPOWER.NS', 'JSFB.NS',
 'JTEKTINDIA.NS', 'JTLIND.NS', 'JUNIPER.NS', 'JWL.NS', 'JYOTHYLAB.NS', 'JYOTISTRUC.NS', 'KABRAEXTRU.NS', 'KALAMANDIR.NS',
 'KAMATHOTEL.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KCP.NS', 'KECL.NS', 'KELLTONTEC.NS', 'KESORAMIND.NS', 'KHADIM.NS',
 'KHAICHEM.NS', 'KILITCH.NS', 'KIOCL.NS', 'KITEX.NS', 'KNRCON.NS', 'KOKUYOCMLN.NS', 'KOLTEPATIL.NS', 'KOPRAN.NS',
 'KOTHARIPET.NS', 'KPEL.NS', 'KPIGREEN.NS', 'KRBL.NS', 'KRISHANA.NS', 'KRITI.NS', 'KRITINUT.NS', 'KRONOX.NS',
 'KROSS.NS', 'KRYSTAL.NS', 'KSOLVES.NS', 'KTKBANK.NS', 'KUANTUM.NS', 'LANDMARK.NS', 'LAOPALA.NS', 'LATENTVIEW.NS',
 'LEMONTREE.NS', 'LIBERTSHOE.NS', 'LIKHITHA.NS', 'LLOYDSENGG.NS', 'LLOYDSENT.NS', 'LTFOODS.NS', 'LXCHEM.NS',
 'MAANALU.NS', 'MADRASFERT.NS', 'MAHLIFE.NS', 'MAHLOG.NS', 'MANAKCOAT.NS', 'MANALIPETC.NS', 'MANAPPURAM.NS',
 'MANBA.NS', 'MANGCHEFER.NS', 'MANINDS.NS', 'MANINFRA.NS', 'MARATHON.NS', 'MARINE.NS', 'MARKSANS.NS', 'MASFIN.NS',
 'MAXESTATES.NS', 'MAXIND.NS', 'MAYURUNIQ.NS', 'MBAPL.NS', 'MBECL.NS', 'MBLINFRA.NS', 'MEDIASSIST.NS', 'MEDICO.NS',
 'MENONBE.NS', 'MHRIL.NS', 'MICEL.NS', 'MIDHANI.NS', 'MINDTECK.NS', 'MMFL.NS', 'MMP.NS', 'MMTC.NS', 'MOIL.NS', 'MOL.NS',
 'MONARCH.NS', 'MOREPENLAB.NS', 'MOTISONS.NS', 'MSPL.NS', 'MSTCLTD.NS', 'MTNL.NS', 'MUFIN.NS', 'MUFTI.NS',
 'MUKANDLTD.NS', 'MUKKA.NS', 'MUNJALAU.NS', 'MUTHOOTMF.NS', 'MVGJL.NS', 'NACLIND.NS', 'NAHARPOLY.NS',
 'NAHARSPING.NS', 'NAVA.NS', 'NAVKARCORP.NS', 'NAVNETEDUL.NS', 'NCC.NS', 'NCLIND.NS', 'NDL.NS', 'NDTV.NS',
 'NECLIFE.NS', 'NELCAST.NS', 'NETWORK18.NS', 'NFL.NS', 'NIITLTD.NS', 'NIITMTS.NS', 'NINSYS.NS', 'NITCO.NS',
 'NITINSPIN.NS', 'NIVABUPA.NS', 'NMDC.NS', 'NOCIL.NS', 'NORTHARC.NS', 'NRAIL.NS', 'NRBBEARING.NS', 'NRL.NS',
 'NSLNISP.NS', 'NUVOCO.NS', 'OAL.NS', 'OMAXE.NS', 'OMINFRAL.NS', 'ONEPOINT.NS', 'ONMOBILE.NS', 'ONWARDTEC.NS',
 'OPTIEMUS.NS', 'ORICONENT.NS', 'ORIENTCEM.NS', 'ORIENTELEC.NS', 'ORIENTHOT.NS', 'ORIENTPPR.NS', 'ORIENTTECH.NS',
 'OSWALAGRO.NS', 'OSWALGREEN.NS', 'PAISALO.NS', 'PAKKA.NS', 'PANACEABIO.NS', 'PANAMAPET.NS', 'PARACABLES.NS',
 'PARADEEP.NS', 'PARAGMILK.NS', 'PARKHOTELS.NS', 'PARSVNATH.NS', 'PATELENG.NS', 'PCBL.NS', 'PCJEWELLER.NS',
 'PDMJEPAPER.NS', 'PDSL.NS', 'PENIND.NS', 'PENINLAND.NS', 'PFOCUS.NS', 'PFS.NS', 'PLASTIBLEN.NS', 'PLATIND.NS',
 'PNBGILTS.NS', 'PNCINFRA.NS', 'PPL.NS', 'PRAKASH.NS', 'PRECAM.NS', 'PRECOT.NS', 'PRECWIRE.NS', 'PREMEXPLN.NS',
 'PREMIERPOL.NS', 'PRICOLLTD.NS', 'PRIMESECU.NS', 'PRINCEPIPE.NS', 'PROZONER.NS', 'PRSMJOHNSN.NS', 'PTC.NS',
 'PTL.NS', 'PURVA.NS', 'PVP.NS', 'PVSL.NS', 'PYRAMID.NS', 'QUICKHEAL.NS', 'RADHIKAJWE.NS', 'RADIANTCMS.NS',
 'RAILTEL.NS', 'RAIN.NS', 'RAJESHEXPO.NS', 'RAJRATAN.NS', 'RAJRILTD.NS', 'RALLIS.NS', 'RAMASTEEL.NS',
 'RAMCOIND.NS', 'RAMCOSYS.NS', 'RAMKY.NS', 'RATNAVEER.NS', 'RBA.NS', 'RBLBANK.NS', 'RBZJEWEL.NS', 'RCF.NS',
 'RCOM.NS', 'REDINGTON.NS', 'REDTAPE.NS', 'REFEX.NS', 'RELAXO.NS', 'RELIGARE.NS', 'RELINFRA.NS', 'RELTD.NS',
 'RENUKA.NS', 'REPCOHOME.NS', 'REPRO.NS', 'RESPONIND.NS', 'RGL.NS', 'RHIM.NS', 'RICOAUTO.NS', 'RISHABH.NS',
 'RITCO.NS', 'RITES.NS', 'RKSWAMY.NS', 'ROHLTD.NS', 'ROTO.NS', 'RPOWER.NS', 'RPPINFRA.NS', 'RPTECH.NS', 'RSWM.NS',
 'RSYSTEMS.NS', 'RTNINDIA.NS', 'RTNPOWER.NS', 'RUBYMILLS.NS', 'RUPA.NS', 'RUSHIL.NS', 'SABTNL.NS', 'SADHNANIQ.NS',
 'SAGCEM.NS', 'SAKAR.NS', 'SAKSOFT.NS', 'SALASAR.NS', 'SAMHI.NS', 'SAMMAANCAP.NS', 'SANDHAR.NS', 'SANDUMA.NS',
 'SANGAMIND.NS', 'SANGHIIND.NS', 'SANGHVIMOV.NS', 'SANSTAR.NS', 'SAPPHIRE.NS', 'SARDAEN.NS', 'SARLAPOLY.NS',
 'SARVESHWAR.NS', 'SASTASUNDR.NS', 'SATIA.NS', 'SATIN.NS', 'SATINDLTD.NS', 'SAURASHCEM.NS', 'SBC.NS', 'SBCL.NS',
 'SBFC.NS', 'SBGLP.NS', 'SCHAND.NS', 'SCI.NS', 'SCILAL.NS', 'SDBL.NS', 'SENCO.NS', 'SEPC.NS', 'SEQUENT.NS',
 'SERVOTECH.NS', 'SESHAPAPER.NS', 'SGIL.NS', 'SHALBY.NS', 'SHALPAINTS.NS', 'SHANTIGEAR.NS', 'SHAREINDIA.NS',
 'SHK.NS', 'SHREDIGCEM.NS', 'SHREEPUSHK.NS', 'SHRIRAMPPS.NS', 'SICALLOG.NS', 'SIGACHI.NS', 'SIGNPOST.NS',
 'SIMPLEXINF.NS', 'SINDHUTRAD.NS', 'SIRCA.NS', 'SIS.NS', 'SKIPPER.NS', 'SKYGOLD.NS', 'SMCGLOBAL.NS', 'SMSPHARMA.NS',
 'SNOWMAN.NS', 'SOLARA.NS', 'SOMANYCERA.NS', 'SONATSOFTW.NS', 'SOTL.NS', 'SOUTHBANK.NS', 'SPANDANA.NS', 'SPARC.NS',
 'SPECIALITY.NS', 'SPENCERS.NS', 'SPIC.NS', 'SPMLINFRA.NS', 'SPORTKING.NS', 'SREEL.NS', 'SRM.NS', 'SSWL.NS',
 'STANLEY.NS', 'STARCEMENT.NS', 'STCINDIA.NS', 'STEELXIND.NS', 'STEL.NS', 'STERTOOLS.NS', 'STLTECH.NS',
 'STYLEBAAZA.NS', 'SUBEXLTD.NS', 'SUKHJITS.NS', 'SULA.NS', 'SUNDARMHLD.NS', 'SUNFLAG.NS', 'SUNTECK.NS',
 'SUPRAJIT.NS', 'SURAJEST.NS', 'SURAJLTD.NS', 'SURYAROSNI.NS', 'SURYODAY.NS', 'SUTLEJTEX.NS', 'SUVEN.NS',
 'SWANENERGY.NS', 'SWSOLAR.NS', 'SYNCOMF.NS', 'SYRMA.NS', 'TAJGVK.NS', 'TALBROAUTO.NS', 'TANLA.NS', 'TARC.NS',
 'TARIL.NS', 'TARSONS.NS', 'TBZ.NS', 'TDPOWERSYS.NS', 'TEXINFRA.NS', 'TEXRAIL.NS', 'TFCILTD.NS', 'THEINVEST.NS',
 'THEMISMED.NS', 'THOMASCOOK.NS', 'TI.NS', 'TIL.NS', 'TIMETECHNO.NS', 'TIRUMALCHM.NS', 'TMB.NS', 'TNPETRO.NS',
 'TNPL.NS', 'TOLINS.NS', 'TPLPLASTEH.NS', 'TRACXN.NS', 'TRANSWORLD.NS', 'TREL.NS', 'TRIDENT.NS', 'TRIVENI.NS',
 'TTML.NS', 'TVSELECT.NS', 'TVSSCS.NS', 'TVTODAY.NS', 'UDAICEMENT.NS', 'UDS.NS', 'UFLEX.NS', 'UGARSUGAR.NS',
 'UGROCAP.NS', 'UJJIVANSFB.NS', 'UNIDT.NS', 'UNIECOM.NS', 'UNIENTER.NS', 'UNIPARTS.NS', 'UNITECH.NS', 'URJA.NS',
 'USHAMART.NS', 'UTKARSHBNK.NS', 'UTTAMSUGAR.NS', 'VAIBHAVGBL.NS', 'VAKRANGEE.NS', 'VALIANTORG.NS', 'VARROC.NS',
 'VASCONEQ.NS', 'VERANDA.NS', 'VERTOZ.NS', 'VGUARD.NS', 'VHLTD.NS', 'VIDHIING.NS', 'VIKASLIFE.NS', 'VINCOFE.NS',
 'VIPIND.NS', 'VISAKAIND.NS', 'VISHNU.NS', 'VLEGOV.NS', 'VLSFINANCE.NS', 'VPRPL.NS', 'VRAJ.NS', 'VRLLOG.NS', 'VSSL.NS',
 'VSTIND.NS', 'VTL.NS', 'WALCHANNAG.NS', 'WANBURY.NS',
 'WCIL.NS', 'WEL.NS', 'WELENT.NS', 'WELSPUNLIV.NS', 'WINDMACHIN.NS', 'WSI.NS', 'WSTCSTPAPR.NS',
 'XCHANGING.NS', 'XTGLOBAL.NS', 'YATHARTH.NS', 'YATRA.NS', 'ZAGGLE.NS', 'ZEEL.NS', 'ZEEMEDIA.NS', 'ZUARI.NS', 'ZUARIIND.NS']
                total = len(intraday_stocks)
                progress_bar = st.progress(0)
                results = []
                
                for i, stock in enumerate(intraday_stocks, 1):
                    result, msg = check_moving_average_conditions(stock)
                    st.write(msg)
                    if result != False:
                        stock_data, output = analyze_stock(stock)
                        st.write(output)
                        if stock_data and not isinstance(stock_data, str):
                            results.append(stock_data)
                    progress_bar.progress(i / total)
                
                if results:
                    df_results = pd.DataFrame(results, columns=names)
                    st.dataframe(df_results)
                    csv_content = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name="intraday_stock_analysis.csv",
                        mime="text/csv"
                    )
                    if st.checkbox("Send Email"):
                        recipient = st.text_input("Recipient Email")
                        sender = st.text_input("Your Email")
                        password = st.text_input("App Password", type="password")
                        if st.button("Send"):
                            subject = f"Intraday Stocks Analysis Report - {date}"
                            body = f"Attached is the intraday analysis report generated on {date}."
                            success, msg = send_email(subject, body, csv_content, recipient, sender, password)
                            st.write(msg)

if __name__ == "__main__":
    main()
