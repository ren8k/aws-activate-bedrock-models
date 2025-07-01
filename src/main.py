import logging

from .bedrock_model_activator import BedrockModelActivator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the Bedrock model activation
    """
    # Set the region for Bedrock operations
    BEDROCK_REGION = 'us-east-1'  # Change this to your preferred region

    try:
        # Initialize the activator
        activator = BedrockModelActivator(region_name=BEDROCK_REGION)

        # Run the activation process
        results = activator.activate_all_models()

        # Print summary
        activator.print_summary(results)

    except Exception as e:
        logger.error(f"Fatal error during model activation: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
