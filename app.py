                expected_columns = getattr(preprocessor, "feature_names_in_", None)
                if expected_columns is not None:
                    expected_columns = list(expected_columns)
                    missing_columns = [c for c in expected_columns if c not in df_input.columns]
                    if missing_columns:
                        raise ValueError(
                            "The input dataset is missing features required by the trained "
                            f"preprocessor: {missing_columns}"
                        )
                    df_input = df_input.reindex(columns=expected_columns)

                # Identify columns assigned to numeric branches of a ColumnTransformer.
                numeric_model_columns = set()
                for transformer_name, transformer, selected_columns in getattr(
                    preprocessor, "transformers_", []
                ):
                    if transformer_name == "remainder" or transformer == "drop":
                        continue

                    if isinstance(selected_columns, slice):
                        selected = list(df_input.columns[selected_columns])
                    elif isinstance(selected_columns, (list, tuple, np.ndarray, pd.Index)):
                        selected = list(selected_columns)
                        if selected and isinstance(selected[0], (bool, np.bool_)):
                            selected = list(df_input.columns[np.asarray(selected, dtype=bool)])
                        elif selected and isinstance(selected[0], (int, np.integer)):
                            selected = [df_input.columns[int(i)] for i in selected]
                    else:
                        selected = [selected_columns] if selected_columns in df_input.columns else []

                    # Numeric pipelines are commonly named "num"/"numeric" or contain
                    # a median/mean imputer, scaler, or other numeric preprocessing step.
                    steps = getattr(transformer, "steps", [])
                    step_names = " ".join(str(name).lower() for name, _ in steps)
                    transformer_text = (
                        str(transformer_name).lower() + " " +
                        transformer.__class__.__name__.lower() + " " + step_names
                    )
                    has_numeric_imputer = any(
                        getattr(step, "strategy", None) in ("mean", "median")
                        for _, step in steps
                    )
                    if (
                        any(token in transformer_text for token in ("numeric", "num", "scal", "standard"))
                        or has_numeric_imputer
                    ):
                        numeric_model_columns.update(selected)

                # Always keep known text/category fields as strings. For other object
                # columns, convert them when every non-empty value is numeric.
                categorical_columns = {
                    "Country", "Agrofor_practice",
                    "curr_EcC_Transplanting", "curr_EcC_total_max",
                    "curr_EcB_income_max", "curr_EcB_tree"
                }

                for col in df_input.columns:
                    series = df_input[col].replace(
                        [np.inf, -np.inf, "inf", "-inf", "Infinity", "-Infinity"], np.nan
                    )

                    must_be_numeric = (
                        col in numeric_model_columns
                        or pd.api.types.is_numeric_dtype(df_clean[col])
                    )

                    if must_be_numeric:
                        converted = pd.to_numeric(series, errors="coerce")
                        # Missing numeric values are replaced with zero only as a
                        # safe fallback; the fitted preprocessor can then transform them.
                        df_input[col] = converted.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    elif col in categorical_columns or not pd.api.types.is_numeric_dtype(series):
                        df_input[col] = series.fillna("Unspecified").astype(str)
                    else:
                        df_input[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)

                # Final guard: no object/string values may enter a numeric model column.
                for col in numeric_model_columns:
                    if col in df_input.columns:
                        df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0.0)

                # Execute pipeline transformation and prediction.
                input_processed = preprocessor.transform(df_input)
                
                if l1_feature_selector is not None:
                    input_processed = l1_feature_selector.transform(input_processed)

                prediction = rf_model.predict(input_processed)[0]

                st.success(f"### 🎉 Predicted Economic Benefits: ${prediction:,.2f} USD (2020)")
            except Exception as ex:
                st.error(f"Error executing model pipeline preprocessing: {ex}")
                with st.expander("Technical diagnostics"):
                    st.write("Input feature data types:")
                    try:
                        st.dataframe(
                            pd.DataFrame({
                                "feature": df_input.columns,
                                "dtype": [str(dtype) for dtype in df_input.dtypes],
                                "sample_value": [
                                    str(df_input.iloc[0][col]) for col in df_input.columns
                                ]
                            }),
                            use_container_width=True
                        )
                    except Exception:
                        st.caption("Input diagnostics are unavailable.")
                    st.caption(
                        "If the error persists, the preprocessor/model files may have been "
                        "trained with a different feature schema or scikit-learn version."
                    )

# --- Footer ---
st.markdown('<div class="footer">Agroforestry Benefits Analytics Dashboard • Built with Streamlit, Pandas & Scikit-Learn</div>', unsafe_allow_html=True)
