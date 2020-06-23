#!/bin/bash
# This script updates the documentation and pushes to the corresponding
# Github repository
# See https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

# Only push when updating main (i.e. new releases)
if [ "$TRAVIS_PULL_REQUEST" != "false" -o "$TRAVIS_BRANCH" != "main" ]; then
    echo "Skipping deploy; just doing a build."
    exit 0
fi

SHA=`git rev-parse --verify HEAD`

# Install sphinx, and build the documentation
pip install sphinx sphinx_rtd_theme
cd docs
make html
cd ..

# Get ssh credential to push to the documentation, using encrypted key
openssl aes-256-cbc -K $encrypted_12c8071d2874_key -iv $encrypted_12c8071d2874_iv -in deploy_key.enc -out deploy_key -d
chmod 600 deploy_key
eval `ssh-agent -s`
ssh-add deploy_key

# Clone the documentation repository
git clone git@github.com:fbpic/fbpic.github.io.git
# Remove the previous `dev` documentation
cd fbpic.github.io
git rm -r ./*
# Copy and add the new documentation
cp -r ../docs/build/html/* ./
git add ./*

# Configure git user and commit changes
git config user.name "Travis CI"
git config user.email "rlehe@normalesup.org"
git commit -m "Deploy to GitHub Pages: ${SHA}" || true
# Push to the repo
git push
