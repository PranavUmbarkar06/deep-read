import json
import os
from google import genai
from google.genai import types

#import logger

MODEL=os.getenv("MODEL", "gemini-2.5-flash")  # Default to gemini-2.5-flash if not set
# Simple evaluation wrapper
def evaluate_summary(source_text: str, summary_text: str) -> dict:
    # Initialize client (picks up GEMINI_API_KEY from environment)
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = f"""
    You are an expert editor evaluating a summary against its source text.
    
    Source Text:
    \"\"\"{source_text}\"\"\"
    
    Generated Summary:
    \"\"\"{summary_text}\"\"\"
    
    Analyze the summary for:
    1. Faithfulness (No hallucinations)
    2. Coverage (Key points included)
    3. Conciseness (No fluff)
    
    Return a JSON object matching this schema:
    {{
        "faithfulness_score": float,
        "coverage_score": float,
        "conciseness_score": float,
        "hallucinations_found": ["list of strings or empty"],
        "missing_key_points": ["list of strings or empty"],
        "feedback":[a list of strings containing tips for regeneration to improve the summary]
    }}
    """
    
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1 # Low temperature for more deterministic, factual evaluation
        )
    )
    #logger.log("Evaluated summary", f"Source Text Length: {len(source_text)} characters, Summary Length: {len(summary_text)} characters, Evaluation Result: {json.loads(response.text)['feedback']}")
    return json.loads(response.text)

# Example Usage
if __name__ == "__main__":
    
    main_text=extract_text_from_pdf("../downloaded_papers/autmotive_nvh_technology/nvh.pdf")
    summary_text="""
        This paper introduces a Bayesian metamodeling approach to predict and assess interior cabin noise in vehicles, accounting for inherent production variability and leveraging existing measurement databases for early-stage design optimization. It combines physical laws with data-driven models to quantify uncertainty in noise, vibration, and harshness (NVH) characteristics.

        ## Core Intent & Problem
        - **What precise problem does this paper target?**
            This paper targets the challenge of accurately assessing and predicting interior cabin noise (Noise, Vibration, and Harshness - NVH) in vehicles, particularly focusing on broadband noises like aerodynamic and tire-road interaction noise. The goal is to develop fast, reliable models that can be used during early design stages to understand and optimize the acoustic signature, which significantly impacts passenger comfort and vehicle perception.

        - **Why do existing solutions fail (according to the authors)?**
            Existing physics-based numerical simulations (e.g., refined 3D finite element models) are computationally intensive and time-consuming, making them impractical for early design exploration. Furthermore, these deterministic models struggle with the high uncertainty arising from manufacturing tolerances, natural variability in material properties, and diverse testing conditions. They cannot provide a measure of uncertainty in the outputs, which is crucial for evaluating design alternatives with varying levels of knowledge.

        ## Key Mechanism & Methodology
        - **How does their proposed solution work step-by-step?**
            1.  **Operating Point (OP) Conditions:** The process begins by identifying typical client usage profiles or driving conditions (e.g., vehicle speed, wheel torque) from collected data, represented as a distribution function.
            2.  **Hybrid Model Formulation (GAMs):** The core of the solution is a "grey-box" metamodel built using Generalized Additive Models (GAMs). These models combine "first-principles" (white-box) physical laws (e.g., sound intensity proportional to v^6 for dipole aerodynamic noise) with data-driven regression fits (black-box) from measurement databases.
            3.  **Basis Functions:** GAMs model the dependency of Sound Pressure Level (SPL) on predictor variables (like speed and frequency) using either polynomial basis functions (simpler, lower computational cost) or Gaussian basis functions (more control over location, width, and scale).
            4.  **Data Selection for Uncertainty Quantification (UQ):** The extensive measurement database is refined by applying categorical variables (e.g., body design, segment, energy type, target market, roof-type, measurement position) to select data relevant to a specific vehicle configuration.
            5.  **Bayesian Framework for UQ:** For aerodynamic noise, a Bayesian approach is applied to the refined dataset. This framework incorporates prior knowledge (domain expertise, physical laws) about the system parameters by defining prior probability distributions. The No-U-Turn Sampler (NUTS), implemented via the PyMC3 library, is used to draw samples from the posterior distribution of these parameters, quantifying their uncertainty.
            6.  **Parametric Bootstrapping for Tire-Road Noise:** For tire-road noise, a non-Bayesian parametric bootstrap algorithm is used. This involves generating simulated data from point estimates of model parameters (obtained via nonlinear least-squares) and then re-estimating parameters from these simulated datasets to assess their dispersion.
            7.  **Model Validation:** Deterministic models are validated using K-fold cross-validation (K=5 or 10). Bayesian models are assessed using MCMC diagnostic tools (rank plots, Gelman-Rubin statistic ˆR) for convergence and Bayesian Leave-One-Out cross-validation (PSIS-LOO-CV) for out-of-sample predictive accuracy.

        - **What datasets or evaluation setups did they use?**
            They used measurement databases containing different noise contributions, specifically aerodynamic noise (from wind-tunnel tests) and tire-pavement interaction noise (rolling noise). The evaluation considered two different vehicle body-types: Sedan and Hatchback. K-fold cross-validation was performed with K=5 and K=10, with 1000 runs. Bayesian models were simulated with 4 chains and 10,000 samples each, with a burn-in of 2,000 samples.

        ## Primary Results & Claims
        - **What are the major quantitative metrics or takeaways?**
            - For deterministic models, K-fold cross-validation showed an R2-CV accuracy of approximately 90% for K=5 (Sedan body-type) with a small variance (6.39e-04).
            - Bayesian Model 1 (BM1, with polynomial basis functions) converged well, with a Gelman-Rubin statistic (ˆR) of 1.0 for all parameters.
            - Bayesian Model 2 (BM2, with Gaussian basis functions and heteroscedasticity) also showed good convergence, with 1.0 ≤ ˆR ≤ 1.06.
            - Based on PSIS-LOO-CV, BM2 (Gaussians) was preferred over BM1 (Polynomials), showing a better LOO value (-1259.2 vs -1367.6) and a lower standard error (22.6 vs 24.1).
            - BM2, despite having a higher number of parameters and being computationally more intensive, captured intricate peaks and physical phenomena better than BM1.
            - For BM2, 0.4% of the data points were identified as potentially influential outliers (ˆk > 0.7), specifically at 4 kHz, 5 kHz, and 6.3 kHz.
            - The parametric bootstrap approach for tire-road noise, while simple, did not capture intricate peaks as effectively as Bayesian methods, suggesting further refinement is needed.

        ## Critical Limitations
        - **What flaws, edge cases, or gaps do the authors acknowledge?**
            - Polynomial basis functions, used in BM1, are global functions of the input variable, which limits their use when the input space changes.
            - BM2, while more accurate, comes at the cost of a higher number of parameters and requires more time to draw samples from the joint distribution.
            - For BM2, a small percentage (0.4%) of data points (at 4 kHz, 5 kHz, and 6.3 kHz) were identified as outliers that the model was not able to fully explain.
            - The non-Bayesian parametric bootstrap algorithm for tire-road noise did not capture intricate peaks in the data, indicating that the model formulation could be further refined. Its accuracy depends on the number of bootstrapped samples and the data-generating mechanism.
            - The paper notes that for overall broadband masking noise estimators, detailed modeling might not always be desired, and BM1 could be preferred for faster computations despite its lower accuracy in capturing fine details.

        """
    result = evaluate_summary(main_text, summary_text)
    print(json.dumps(result, indent=2))