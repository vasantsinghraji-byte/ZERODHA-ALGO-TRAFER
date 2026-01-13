"""
ONE-CLICK TRADING TERMINAL
===========================
This script does EVERYTHING you need:
1. Login to Zerodha (auto-saves token)
2. Show live market prices
3. Let you trade with simple buttons
4. Track your P&L

Just double-click this file every morning!
"""

import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from kiteconnect import KiteConnect
import yaml
import threading
import time
from datetime import datetime

# Your credentials
API_KEY = "dt3y62zval0osg5h"
API_SECRET = "utmejmwjcdx3kogb90sw2dqrxv4qp6q9"

class SimpleTradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Trading Terminal")
        self.root.geometry("900x700")

        self.kite = None
        self.access_token = None
        self.positions = {}
        self.pnl = 0

        # Create UI
        self.create_ui()

        # Try to load saved token
        self.try_load_token()

    def create_ui(self):
        # Title
        title = tk.Label(self.root, text="🚀 ZERODHA TRADING TERMINAL",
                        font=("Arial", 20, "bold"), bg="#1e88e5", fg="white", pady=10)
        title.pack(fill=tk.X)

        # Login Section
        login_frame = tk.LabelFrame(self.root, text="Login", font=("Arial", 12, "bold"), padx=10, pady=10)
        login_frame.pack(fill=tk.X, padx=10, pady=5)

        self.login_status = tk.Label(login_frame, text="Not logged in", fg="red", font=("Arial", 10))
        self.login_status.pack()

        login_btn = tk.Button(login_frame, text="🔐 LOGIN TO ZERODHA", command=self.login,
                             bg="#4caf50", fg="white", font=("Arial", 12, "bold"), padx=20, pady=10)
        login_btn.pack(pady=5)

        # Market Watch
        market_frame = tk.LabelFrame(self.root, text="Live Market Prices",
                                     font=("Arial", 12, "bold"), padx=10, pady=10)
        market_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # RELIANCE
        rel_frame = tk.Frame(market_frame, relief=tk.RAISED, borderwidth=2, bg="#e3f2fd")
        rel_frame.pack(fill=tk.X, pady=5)

        tk.Label(rel_frame, text="RELIANCE", font=("Arial", 14, "bold"), bg="#e3f2fd").pack(side=tk.LEFT, padx=10)
        self.rel_price = tk.Label(rel_frame, text="₹ 0.00", font=("Arial", 16), bg="#e3f2fd")
        self.rel_price.pack(side=tk.LEFT, padx=20)

        tk.Button(rel_frame, text="🟢 BUY", command=lambda: self.trade("RELIANCE", "BUY"),
                 bg="#4caf50", fg="white", font=("Arial", 12, "bold"), padx=15).pack(side=tk.RIGHT, padx=5)
        tk.Button(rel_frame, text="🔴 SELL", command=lambda: self.trade("RELIANCE", "SELL"),
                 bg="#f44336", fg="white", font=("Arial", 12, "bold"), padx=15).pack(side=tk.RIGHT, padx=5)

        tk.Label(rel_frame, text="Qty:", bg="#e3f2fd").pack(side=tk.RIGHT)
        self.rel_qty = tk.Spinbox(rel_frame, from_=1, to=1000, width=5, font=("Arial", 12))
        self.rel_qty.pack(side=tk.RIGHT, padx=5)

        # TCS
        tcs_frame = tk.Frame(market_frame, relief=tk.RAISED, borderwidth=2, bg="#fff3e0")
        tcs_frame.pack(fill=tk.X, pady=5)

        tk.Label(tcs_frame, text="TCS", font=("Arial", 14, "bold"), bg="#fff3e0").pack(side=tk.LEFT, padx=10)
        self.tcs_price = tk.Label(tcs_frame, text="₹ 0.00", font=("Arial", 16), bg="#fff3e0")
        self.tcs_price.pack(side=tk.LEFT, padx=20)

        tk.Button(tcs_frame, text="🟢 BUY", command=lambda: self.trade("TCS", "BUY"),
                 bg="#4caf50", fg="white", font=("Arial", 12, "bold"), padx=15).pack(side=tk.RIGHT, padx=5)
        tk.Button(tcs_frame, text="🔴 SELL", command=lambda: self.trade("TCS", "SELL"),
                 bg="#f44336", fg="white", font=("Arial", 12, "bold"), padx=15).pack(side=tk.RIGHT, padx=5)

        tk.Label(tcs_frame, text="Qty:", bg="#fff3e0").pack(side=tk.RIGHT)
        self.tcs_qty = tk.Spinbox(tcs_frame, from_=1, to=1000, width=5, font=("Arial", 12))
        self.tcs_qty.pack(side=tk.RIGHT, padx=5)

        # Account Summary
        account_frame = tk.LabelFrame(self.root, text="Your Account",
                                      font=("Arial", 12, "bold"), padx=10, pady=10)
        account_frame.pack(fill=tk.X, padx=10, pady=5)

        summary_frame = tk.Frame(account_frame)
        summary_frame.pack()

        tk.Label(summary_frame, text="P&L:", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10)
        self.pnl_label = tk.Label(summary_frame, text="₹ 0.00", font=("Arial", 14), fg="green")
        self.pnl_label.grid(row=0, column=1, padx=10)

        tk.Label(summary_frame, text="Cash:", font=("Arial", 12, "bold")).grid(row=0, column=2, padx=10)
        self.cash_label = tk.Label(summary_frame, text="₹ 1,00,000.00", font=("Arial", 14))
        self.cash_label.grid(row=0, column=3, padx=10)

        # Positions
        pos_frame = tk.LabelFrame(self.root, text="Your Positions",
                                  font=("Arial", 12, "bold"), padx=10, pady=10)
        pos_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.positions_text = tk.Text(pos_frame, height=8, font=("Arial", 10))
        self.positions_text.pack(fill=tk.BOTH, expand=True)

    def try_load_token(self):
        try:
            with open('config/secrets.yaml', 'r') as f:
                secrets = yaml.safe_load(f)
                token = secrets.get('zerodha_access_token')

                if token and token != 'your_access_token_here':
                    self.access_token = token
                    self.kite = KiteConnect(api_key=API_KEY)
                    self.kite.set_access_token(token)

                    # Test if token works
                    try:
                        self.kite.margins()
                        self.login_status.config(text="✅ Logged in!", fg="green")
                        self.start_price_updates()
                        return True
                    except:
                        pass
        except:
            pass

        return False

    def login(self):
        # Open login window
        login_window = tk.Toplevel(self.root)
        login_window.title("Login to Zerodha")
        login_window.geometry("600x400")

        tk.Label(login_window, text="Login to Zerodha", font=("Arial", 16, "bold")).pack(pady=10)

        instructions = tk.Text(login_window, height=10, wrap=tk.WORD, font=("Arial", 10))
        instructions.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        instructions.insert(1.0, """
STEPS TO LOGIN:

1. Click 'Open Zerodha Login' button below
2. Login with your Zerodha credentials
3. After login, you'll see an ERROR page - THIS IS NORMAL!
4. Copy the COMPLETE URL from your browser
5. Paste it in the box below
6. Click 'Submit'

The URL will look like:
http://127.0.0.1/?request_token=XXXXX&action=login&status=success
        """)
        instructions.config(state=tk.DISABLED)

        tk.Button(login_window, text="🌐 Open Zerodha Login",
                 command=lambda: webbrowser.open(f"https://kite.zerodha.com/connect/login?api_key={API_KEY}&v=3"),
                 bg="#ff9800", fg="white", font=("Arial", 12, "bold"), pady=5).pack(pady=5)

        tk.Label(login_window, text="Paste the complete URL here:", font=("Arial", 10)).pack()

        url_entry = tk.Entry(login_window, width=60, font=("Arial", 10))
        url_entry.pack(pady=5)

        def submit_url():
            url = url_entry.get().strip()
            if 'request_token=' in url:
                try:
                    request_token = url.split('request_token=')[1].split('&')[0]

                    # Generate access token
                    kite = KiteConnect(api_key=API_KEY)
                    data = kite.generate_session(request_token, api_secret=API_SECRET)
                    self.access_token = data["access_token"]

                    # Save token
                    with open('config/secrets.yaml', 'r') as f:
                        secrets = yaml.safe_load(f)
                    secrets['zerodha_access_token'] = self.access_token
                    with open('config/secrets.yaml', 'w') as f:
                        yaml.dump(secrets, f, default_flow_style=False, sort_keys=False)

                    # Set token
                    self.kite = kite
                    self.kite.set_access_token(self.access_token)

                    self.login_status.config(text="✅ Logged in!", fg="green")
                    login_window.destroy()
                    messagebox.showinfo("Success", "Login successful! Starting live prices...")
                    self.start_price_updates()

                except Exception as e:
                    messagebox.showerror("Error", f"Login failed: {e}\n\nTry again with a fresh URL!")
            else:
                messagebox.showerror("Error", "Invalid URL! Please paste the complete URL from browser.")

        tk.Button(login_window, text="✅ Submit", command=submit_url,
                 bg="#4caf50", fg="white", font=("Arial", 12, "bold"), pady=5).pack(pady=10)

    def start_price_updates(self):
        def update_prices():
            while True:
                if self.kite:
                    try:
                        # Get live quotes
                        quotes = self.kite.quote(["NSE:RELIANCE", "NSE:TCS"])

                        rel_price = quotes["NSE:RELIANCE"]["last_price"]
                        tcs_price = quotes["NSE:TCS"]["last_price"]

                        self.rel_price.config(text=f"₹ {rel_price:.2f}")
                        self.tcs_price.config(text=f"₹ {tcs_price:.2f}")

                        # Update positions
                        self.update_positions()

                    except Exception as e:
                        print(f"Price update error: {e}")

                time.sleep(2)

        thread = threading.Thread(target=update_prices, daemon=True)
        thread.start()

    def trade(self, symbol, transaction_type):
        if not self.kite:
            messagebox.showerror("Error", "Please login first!")
            return

        qty = int(self.rel_qty.get()) if symbol == "RELIANCE" else int(self.tcs_qty.get())

        try:
            # Place order (PAPER TRADING - modify for real trading)
            # For paper trading, just simulate

            color = "🟢" if transaction_type == "BUY" else "🔴"
            message = f"{color} {transaction_type} {qty} {symbol}\n\n"
            message += "✅ Order placed successfully (PAPER MODE)\n\n"
            message += "Note: This is PAPER TRADING\n"
            message += "To enable real trading, modify the code"

            messagebox.showinfo("Order Placed", message)

            # Track position
            if symbol not in self.positions:
                self.positions[symbol] = {"qty": 0, "avg_price": 0}

            quotes = self.kite.quote([f"NSE:{symbol}"])
            current_price = quotes[f"NSE:{symbol}"]["last_price"]

            if transaction_type == "BUY":
                old_qty = self.positions[symbol]["qty"]
                old_avg = self.positions[symbol]["avg_price"]
                new_qty = old_qty + qty
                new_avg = ((old_qty * old_avg) + (qty * current_price)) / new_qty if new_qty > 0 else current_price
                self.positions[symbol] = {"qty": new_qty, "avg_price": new_avg}
            else:
                self.positions[symbol]["qty"] -= qty

            self.update_positions()

        except Exception as e:
            messagebox.showerror("Error", f"Order failed: {e}")

    def update_positions(self):
        self.positions_text.delete(1.0, tk.END)

        if not self.positions or all(p["qty"] == 0 for p in self.positions.values()):
            self.positions_text.insert(1.0, "No open positions")
            return

        total_pnl = 0

        for symbol, pos in self.positions.items():
            if pos["qty"] != 0:
                try:
                    quotes = self.kite.quote([f"NSE:{symbol}"])
                    current_price = quotes[f"NSE:{symbol}"]["last_price"]
                    pnl = (current_price - pos["avg_price"]) * pos["qty"]
                    total_pnl += pnl

                    pnl_str = f"+₹{pnl:.2f}" if pnl >= 0 else f"-₹{abs(pnl):.2f}"
                    color_tag = "green" if pnl >= 0 else "red"

                    text = f"{symbol}: {pos['qty']} @ ₹{pos['avg_price']:.2f} | LTP: ₹{current_price:.2f} | P&L: {pnl_str}\n"
                    self.positions_text.insert(tk.END, text)
                    self.positions_text.tag_add(color_tag, f"{self.positions_text.index(tk.END)}-2l", f"{self.positions_text.index(tk.END)}-1l")
                    self.positions_text.tag_config("green", foreground="green")
                    self.positions_text.tag_config("red", foreground="red")
                except:
                    pass

        pnl_text = f"₹ {total_pnl:.2f}"
        pnl_color = "green" if total_pnl >= 0 else "red"
        self.pnl_label.config(text=pnl_text, fg=pnl_color)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleTradingApp(root)
    root.mainloop()
