            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.backtest_content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store canvas reference
        self.backtest_canvas = canvas

        # Initial placeholder
        self.show_backtest_placeholder()

    def show_backtest_placeholder(self):
        """Show placeholder when no backtest results"""
        # Clear existing content
        for widget in self.backtest_content_frame.winfo_children():
            widget.destroy()

        placeholder_frame = tk.Frame(
            self.backtest_content_frame,
            bg=COLORS['bg_secondary'],
            relief=tk.RAISED,
            borderwidth=2
        )
        placeholder_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=50)

        tk.Label(
            placeholder_frame,
            text="📊 No Backtest Results Yet",
            font=("Arial", 18, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary']
        ).pack(pady=30)

        tk.Label(
            placeholder_frame,
            text="Run a backtest to see comprehensive results here.\n\n"
                 "Click 'Run New Backtest' button above to get started.",
            font=("Arial", 12),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary'],
            justify=tk.CENTER
        ).pack(pady=20)

        tk.Button(
            placeholder_frame,
            text="▶ Run Your First Backtest",
            command=self.run_backtest,
            bg=COLORS['neon_blue'],
            fg=COLORS['text_primary'],
            font=("Arial", 12, "bold"),
            padx=30,
            pady=15,
            cursor='hand2'
        ).pack(pady=30)

    def create_scanner_tab(self):
        """Market scanner tab"""
        tab = tk.Frame(self.notebook, bg=COLORS['bg_primary'])
        self.notebook.add(tab, text="  🔍 Scanner  ")

        # Header
        header_frame = tk.Frame(tab, bg=COLORS['bg_primary'])
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            header_frame,
            text="🔍 Market Scanner",
            font=("Arial", 16, "bold"),
            bg=COLORS['bg_primary'],
            fg=COLORS['neon_blue']
        ).pack(side=tk.LEFT)

        # Scanner type selector
        control_frame = tk.Frame(tab, bg=COLORS['bg_secondary'], padx=15, pady=10)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            control_frame,
            text="Scanner Type:",
            font=("Arial", 11),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)

        self.scanner_type_var = tk.StringVar(value='volume')
        scanner_types = [
            ('Volume Surge', 'volume'),
            ('Pre-Market Gaps', 'premarket'),
            ('Breakouts', 'breakout'),
            ('Pattern Recognition', 'pattern')
        ]

        for text, value in scanner_types:
            tk.Radiobutton(
                control_frame,
                text=text,
                variable=self.scanner_type_var,
                value=value,
                bg=COLORS['bg_secondary'],
                fg=COLORS['text_primary'],
                selectcolor=COLORS['bg_tertiary'],
                font=("Arial", 10)
            ).pack(side=tk.LEFT, padx=10)

        # Run scan button
        tk.Button(
            control_frame,
            text="▶ Run Scan",
            command=self.run_scanner,
            bg=COLORS['neon_green'],
            fg=COLORS['text_primary'],
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.RIGHT, padx=10)

        # Results frame
        results_container = tk.Frame(tab, bg=COLORS['bg_primary'])
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Results label
        self.scanner_results_label = tk.Label(
            results_container,
            text="Click 'Run Scan' to find trading opportunities...",
            font=("Arial", 11),
            bg=COLORS['bg_primary'],
            fg=COLORS['text_secondary']
        )
        self.scanner_results_label.pack(pady=10)

        # Results table
        table_frame = tk.Frame(results_container, bg=COLORS['bg_secondary'])
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        vsb = tk.Scrollbar(table_frame, orient="vertical")
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        hsb = tk.Scrollbar(table_frame, orient="horizontal")
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # Treeview for results - ENHANCED WITH ENTRY/SL/TARGET
        columns = ('Symbol', 'Signal', 'Entry', 'SL', 'Target', 'R:R', 'Score')
        self.scanner_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            height=15
        )

        # Column headings
        self.scanner_tree.heading('Symbol', text='Symbol')
        self.scanner_tree.heading('Signal', text='Signal')
        self.scanner_tree.heading('Entry', text='Entry')
        self.scanner_tree.heading('SL', text='Stop Loss')
        self.scanner_tree.heading('Target', text='Target')
        self.scanner_tree.heading('R:R', text='Risk:Reward')
        self.scanner_tree.heading('Score', text='Score')

        # Column widths
        self.scanner_tree.column('Symbol', width=100)
        self.scanner_tree.column('Signal', width=180)
        self.scanner_tree.column('Entry', width=90)
        self.scanner_tree.column('SL', width=90)
        self.scanner_tree.column('Target', width=90)
        self.scanner_tree.column('R:R', width=70)
        self.scanner_tree.column('Score', width=70)

        self.scanner_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb.config(command=self.scanner_tree.yview)
        hsb.config(command=self.scanner_tree.xview)

        # Action buttons
        action_frame = tk.Frame(results_container, bg=COLORS['bg_primary'])
        action_frame.pack(fill=tk.X, pady=10)

        tk.Button(
            action_frame,
            text="➕ Add Selected to Watchlist",
            command=self.add_scanner_result_to_watchlist,
            bg=COLORS['neon_blue'],
            fg=COLORS['text_primary'],
            font=("Arial", 10, "bold"),
            padx=15,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            action_frame,
            text="📊 Export Results",
            command=self.export_scanner_results,
            bg=COLORS['neon_purple'],
            fg=COLORS['text_primary'],
            font=("Arial", 10, "bold"),
            padx=15,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)

    def create_portfolio_tab(self):
        """Portfolio & Correlation Analysis tab"""
        tab = tk.Frame(self.notebook, bg=COLORS['bg_primary'])
        self.notebook.add(tab, text="  📊 Portfolio  ")

        if not PORTFOLIO_AVAILABLE:
            error_frame = tk.Frame(tab, bg=COLORS['bg_primary'])
            error_frame.pack(expand=True)

            tk.Label(
                error_frame,
                text="⚠ Portfolio Analysis Module Not Available",
                font=("Arial", 16, "bold"),
                bg=COLORS['bg_primary'],
                fg=COLORS['neon_amber']
            ).pack(pady=10)

            tk.Label(
                error_frame,
                text="The portfolio analysis package failed to load.\n\n"
                     "This usually means:\n"
                     "1. Missing scipy dependency (run: pip install scipy)\n"
                     "2. Portfolio package has import errors\n\n"
                     "Check the console output for detailed error messages.",
                font=("Arial", 11),
                bg=COLORS['bg_primary'],
                fg=COLORS['text_secondary'],
                justify=tk.LEFT
            ).pack(pady=10, padx=20)

            # Debug button
            def show_debug_info():
                try:
                    from portfolio import CorrelationAnalyzer
                    messagebox.showinfo("Debug", "Portfolio module imports successfully!")
                except Exception as e:
                    messagebox.showerror("Debug Error", f"Import failed:\n{str(e)}")

            tk.Button(
                error_frame,
                text="🔧 Test Portfolio Import",
                command=show_debug_info,
                bg=COLORS['neon_blue'],
                fg=COLORS['text_primary'],
                font=("Arial", 10, "bold"),
                padx=20,
                pady=8
            ).pack(pady=10)

            return

        # Header
        header_frame = tk.Frame(tab, bg=COLORS['bg_primary'])
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            header_frame,
            text="📊 Portfolio & Correlation Analysis",
            font=("Arial", 16, "bold"),
            bg=COLORS['bg_primary'],
            fg=COLORS['neon_purple']
        ).pack(side=tk.LEFT)

        # Create sub-notebook for portfolio tools
        portfolio_notebook = ttk.Notebook(tab)
        portfolio_notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # ===== SUB-TAB 1: CORRELATION ANALYSIS =====
        self.create_correlation_tab(portfolio_notebook)

        # ===== SUB-TAB 2: PORTFOLIO OPTIMIZATION =====
        self.create_optimization_tab(portfolio_notebook)

        # ===== SUB-TAB 3: SECTOR ANALYSIS =====
        self.create_sector_analysis_tab(portfolio_notebook)

        # ===== SUB-TAB 4: RISK ANALYSIS =====
        self.create_risk_analysis_tab(portfolio_notebook)

    def create_correlation_tab(self, parent_notebook):
        """Correlation & Covariance Analysis sub-tab"""
        tab = tk.Frame(parent_notebook, bg=COLORS['bg_primary'])
        parent_notebook.add(tab, text="  Correlation  ")

        # Control panel
        control_frame = tk.Frame(tab, bg=COLORS['bg_secondary'], padx=15, pady=10)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        # Symbol input
        tk.Label(
            control_frame,
            text="Symbols (comma-separated):",
            font=("Arial", 11),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)

        self.corr_symbols_entry = tk.Entry(
            control_frame,
            font=("Arial", 10),
            width=40,
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary']
        )
        self.corr_symbols_entry.pack(side=tk.LEFT, padx=10)
        self.corr_symbols_entry.insert(0, "RELIANCE, TCS, INFY, HDFC, ICICIBANK")

        # Load from Zerodha Portfolio button
        tk.Button(
            control_frame,
            text="📥 Load My Portfolio",
            command=self.load_zerodha_portfolio_symbols,
            bg=COLORS['neon_blue'],
            fg=COLORS['text_primary'],
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)

        # Correlation method
        tk.Label(
            control_frame,
            text="Method:",
            font=("Arial", 11),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)

        self.corr_method_var = tk.StringVar(value='pearson')
        for method in ['pearson', 'spearman', 'kendall']:
            tk.Radiobutton(
                control_frame,
                text=method.capitalize(),
                variable=self.corr_method_var,
                value=method,
                bg=COLORS['bg_secondary'],
                fg=COLORS['text_primary'],
                selectcolor=COLORS['bg_tertiary'],
                font=("Arial", 10)
            ).pack(side=tk.LEFT, padx=5)

        # Calculate button
        tk.Button(
            control_frame,
            text="▶ Calculate Correlation",
            command=self.calculate_correlation,
            bg=COLORS['neon_purple'],
            fg=COLORS['text_primary'],
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.RIGHT, padx=10)

        # Results area
        results_frame = tk.Frame(tab, bg=COLORS['bg_primary'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left: Correlation Matrix Display
        left_frame = tk.Frame(results_frame, bg=COLORS['bg_secondary'], padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            left_frame,
            text="Correlation Matrix",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['neon_blue']
        ).pack(pady=5)

        # Scrollable text for correlation matrix
        self.corr_matrix_text = scrolledtext.ScrolledText(
            left_frame,
            font=("Consolas", 9),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            height=20,
            wrap=tk.NONE
        )
        self.corr_matrix_text.pack(fill=tk.BOTH, expand=True)

        # Right: Key Insights
        right_frame = tk.Frame(results_frame, bg=COLORS['bg_secondary'], padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right_frame,
            text="Key Insights",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['neon_green']
        ).pack(pady=5)

        self.corr_insights_text = scrolledtext.ScrolledText(
            right_frame,
            font=("Arial", 10),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            height=20,
            wrap=tk.WORD
        )
        self.corr_insights_text.pack(fill=tk.BOTH, expand=True)

    def create_optimization_tab(self, parent_notebook):
        """Portfolio Optimization sub-tab"""
        tab = tk.Frame(parent_notebook, bg=COLORS['bg_primary'])
        parent_notebook.add(tab, text="  Optimization  ")

        # Control panel
        control_frame = tk.Frame(tab, bg=COLORS['bg_secondary'], padx=15, pady=10)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        # Optimization method
        tk.Label(
            control_frame,
            text="Method:",
            font=("Arial", 11),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)

        self.opt_method_var = tk.StringVar(value='max_sharpe')
        opt_methods = [
            ('Max Sharpe', 'max_sharpe'),
            ('Min Volatility', 'min_volatility'),
            ('Risk Parity', 'risk_parity'),
            ('Black-Litterman', 'black_litterman')
        ]

        for text, value in opt_methods:
            tk.Radiobutton(
                control_frame,
                text=text,
                variable=self.opt_method_var,
                value=value,
                bg=COLORS['bg_secondary'],
                fg=COLORS['text_primary'],
                selectcolor=COLORS['bg_tertiary'],
                font=("Arial", 10)
            ).pack(side=tk.LEFT, padx=10)

        # Optimize button
        tk.Button(
            control_frame,
            text="▶ Optimize Portfolio",
            command=self.optimize_portfolio,
            bg=COLORS['neon_green'],
            fg=COLORS['text_primary'],
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.RIGHT, padx=10)

        # Results area
        results_frame = tk.Frame(tab, bg=COLORS['bg_primary'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Portfolio weights
        weights_frame = tk.Frame(results_frame, bg=COLORS['bg_secondary'], padx=10, pady=10)
        weights_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            weights_frame,
            text="Optimized Portfolio Weights",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['neon_purple']
        ).pack(pady=5)

        # Portfolio stats
        stats_frame = tk.Frame(weights_frame, bg=COLORS['bg_tertiary'], padx=10, pady=10)
        stats_frame.pack(fill=tk.X, pady=10)

        self.opt_stats_labels = {}
        for stat in ['Expected Return', 'Volatility', 'Sharpe Ratio']:
            row = tk.Frame(stats_frame, bg=COLORS['bg_tertiary'])
            row.pack(fill=tk.X, pady=3)
            tk.Label(
                row,
                text=f"{stat}:",
                font=("Arial", 10, "bold"),
                bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary'],
                width=15,
                anchor='w'
            ).pack(side=tk.LEFT, padx=5)
            label = tk.Label(
                row,
                text="--",
                font=("Arial", 10),
                bg=COLORS['bg_tertiary'],
                fg=COLORS['neon_blue']
            )
            label.pack(side=tk.LEFT, padx=5)
            self.opt_stats_labels[stat] = label

        # Weights table
        self.opt_weights_text = scrolledtext.ScrolledText(
            weights_frame,
            font=("Consolas", 10),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            height=15
        )
        self.opt_weights_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def create_sector_analysis_tab(self, parent_notebook):
        """Sector Exposure Analysis sub-tab"""
        tab = tk.Frame(parent_notebook, bg=COLORS['bg_primary'])
        parent_notebook.add(tab, text="  Sector Analysis  ")

        # Control panel
        control_frame = tk.Frame(tab, bg=COLORS['bg_secondary'], padx=15, pady=10)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            control_frame,
            text="Analyze sector concentration and diversification",
            font=("Arial", 11),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary']
        ).pack(side=tk.LEFT, padx=5)

        # Analyze button
        tk.Button(
            control_frame,
            text="▶ Analyze Sectors",
            command=self.analyze_sectors,
            bg=COLORS['neon_amber'],
            fg=COLORS['text_primary'],
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.RIGHT, padx=10)

        # Results area
        results_frame = tk.Frame(tab, bg=COLORS['bg_primary'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Metrics frame
        metrics_frame = tk.Frame(results_frame, bg=COLORS['bg_secondary'], padx=10, pady=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            metrics_frame,
            text="Concentration Metrics",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['neon_blue']
        ).pack(pady=5)

        self.sector_metrics_labels = {}
        for metric in ['HHI (Concentration)', 'Effective # Sectors', 'Diversification Score']:
            row = tk.Frame(metrics_frame, bg=COLORS['bg_tertiary'], padx=10, pady=5)
            row.pack(fill=tk.X, pady=3)
            tk.Label(
                row,
                text=f"{metric}:",
                font=("Arial", 10, "bold"),
                bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary'],
                width=25,
                anchor='w'
            ).pack(side=tk.LEFT, padx=5)
            label = tk.Label(
                row,
                text="--",
                font=("Arial", 10),
                bg=COLORS['bg_tertiary'],
                fg=COLORS['neon_green']
            )
            label.pack(side=tk.LEFT, padx=5)
            self.sector_metrics_labels[metric] = label

        # Sector breakdown
        breakdown_frame = tk.Frame(results_frame, bg=COLORS['bg_secondary'], padx=10, pady=10)
        breakdown_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            breakdown_frame,
            text="Sector Breakdown",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['neon_purple']
        ).pack(pady=5)

        self.sector_breakdown_text = scrolledtext.ScrolledText(
            breakdown_frame,
            font=("Consolas", 10),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            height=15
        )
        self.sector_breakdown_text.pack(fill=tk.BOTH, expand=True)

    def create_risk_analysis_tab(self, parent_notebook):
        """Risk Analysis sub-tab (Beta, Alpha, Hedging)"""
        tab = tk.Frame(parent_notebook, bg=COLORS['bg_primary'])
        parent_notebook.add(tab, text="  Risk Analysis  ")

        # Control panel
        control_frame = tk.Frame(tab, bg=COLORS['bg_secondary'], padx=15, pady=10)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        # Benchmark selection
        tk.Label(
            control_frame,
            text="Benchmark:",
            font=("Arial", 11),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)

        self.benchmark_var = tk.StringVar(value='NIFTY50')
        benchmarks = ['NIFTY50', 'BANKNIFTY', 'SENSEX']
        benchmark_menu = ttk.Combobox(
            control_frame,
            textvariable=self.benchmark_var,
            values=benchmarks,
            width=15,
            state='readonly'
        )
        benchmark_menu.pack(side=tk.LEFT, padx=10)

        # Analyze button
        tk.Button(
            control_frame,
            text="▶ Calculate Beta/Alpha",
            command=self.calculate_beta_alpha,
            bg=COLORS['neon_red'],
            fg=COLORS['text_primary'],
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.RIGHT, padx=10)

        # Results area
        results_frame = tk.Frame(tab, bg=COLORS['bg_primary'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Beta/Alpha metrics
        metrics_frame = tk.Frame(results_frame, bg=COLORS['bg_secondary'], padx=10, pady=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            metrics_frame,
            text="Portfolio Risk Metrics",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['neon_red']
        ).pack(pady=5)

        self.risk_metrics_text = scrolledtext.ScrolledText(
            metrics_frame,
            font=("Consolas", 10),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            height=20
        )
        self.risk_metrics_text.pack(fill=tk.BOTH, expand=True, pady=10)

        # Hedging recommendations
        hedge_frame = tk.Frame(results_frame, bg=COLORS['bg_secondary'], padx=10, pady=10)
        hedge_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        tk.Label(
            hedge_frame,
            text="Hedging Recommendations",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_secondary'],
            fg=COLORS['neon_amber']
        ).pack(pady=5)

        self.hedge_text = scrolledtext.ScrolledText(
            hedge_frame,
            font=("Arial", 10),
            bg=COLORS['bg_tertiary'],
            fg=COLORS['text_primary'],
            height=10
        )
        self.hedge_text.pack(fill=tk.BOTH, expand=True)

    # Portfolio analysis methods
    def calculate_correlation(self):
        """Calculate correlation matrix for selected symbols"""
        if not PORTFOLIO_AVAILABLE: