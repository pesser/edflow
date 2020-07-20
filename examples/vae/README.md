# Example

* explore dataset using `edexplore`
* on mnist
![assets/edexplore_mnist.gif](assets/edexplore_mnist.gif)

* on deepfashion, which has additional annotations such as segmentation and IUV flow.
![assets/edexplore_df.gif](assets/edexplore_df.gif)

```
export STREAMLIT_SERVER_PORT=8080
edexplore -b vae/config_explore.yaml
```


* train and evaluate model
```
edflow -b vae/config.yaml -t # abort at some point
edflow -b vae/config.yaml -p logs/xxx # will also trigger evaluation
```

* will generate FID outputs

![assets/FID_eval.png](assets/FID_eval.png)


