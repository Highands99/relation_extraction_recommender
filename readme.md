# README - Setup e Utilizzo del Suggestion Server per Inception

## Fase 1 - Installare Inception e predisporre il progetto

1. **Scarica Inception**
   - [Inception v35.5](https://github.com/inception-project/inception/releases/download/inception-35.5/inception-app-webapp-35.5-standalone.jar)
2. **Esegui Inception**
   ```sh
   java -jar inception-app-webapp-35.5-standalone.jar
   ```
3. **Accedi a Inception**
   - Apri il browser e vai su [http://localhost:8080/](http://localhost:8080/)
   - Effettua la registrazione di un utente admin
4. **Crea un nuovo progetto**
   - Scegli il tipo: `Basic Annotation (Span/Relation)`
5. **Configura il suggeritore**
   - Vai su `Settings -> Recommenders -> Create`
   - Imposta:
     - `Layer`: Relation
     - `Feature`: Label
     - `Tool`: Remote Classifier
     - `Remote URL`: `http://127.0.0.1:5000/wikidata_rel`

## Fase 2 - Importare il dataset in Inception

Segui la guida disponibile qui: [convert2CAS](https://github.com/Highands99/convert2CAS)

## Fase 3 - Installare ed eseguire il server

1. **Scaricare la repository**
2. **Installare le dipendenze nell'ambiente virtuale**
   ```sh
   pipenv install -r requirements.txt
   ```
3. **Entrare nell'ambiente virtuale**
   ```sh
   pipenv shell
   ```
4. **Eseguire il server**
   ```sh
   python wsgi.py
   ```

## Fase 4 - Prova il sistema

1. **Apri un documento** nel progetto creato precedentemente in Inception
2. **Verifica il suggeritore**
   - Inception invia automaticamente una richiesta al server all'apertura del documento
   - Per forzare l'aggiornamento manualmente:
     - Seleziona la tab `Recommender`
     - Clicca su `Retrain`
3. **Verifica i suggerimenti generati**
   - Se Inception non mostra i suggerimenti:
     - Vai nella tab `Recommender`
     - Clicca sull'icona dell'ingranaggio
     - Spunta `Show hidden suggestion`
     - Refresha il documento con l'icona di aggiornamento o premi `F5`
4. **Controlla le relazioni suggerite**
   - Dovresti vedere una relazione aggiuntiva con collegamenti grigi (anziché blu) e con un'emoji di robot all'inizio

---
### Note
- Assicurati che il server sia attivo prima di aprire un documento in Inception
- Se riscontri problemi, verifica i log del server per eventuali errori

